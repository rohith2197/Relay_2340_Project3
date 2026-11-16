"""Offline US 6 tests: real preprocessing/storage/retrieval, mocked HTTP and DNS."""
import io
import socket
from unittest.mock import MagicMock, patch

from django.core.management import call_command, CommandError
from django.test import SimpleTestCase, TestCase, override_settings

from knowledge.ingestion import ingest_document
from knowledge.models import Document
from knowledge.scraping import MAX_BYTES, ScrapingError, _connect, fetch_url
from rag import retrieval


URL = "https://example.org/appeals"
TEXT = ("FEMA appeals: send supporting documents within 60 days of the decision letter. "
        "Include your application number and disaster number on each page. "
        "Keep copies of the documents submitted with your appeal.")
HTML = f"<html><nav>Menu</nav><script>bad script</script><main><p>{TEXT}</p></main></html>"
PUBLIC = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


def response(body=HTML.encode(), status=200, **headers):
    reply = MagicMock()
    reply.status = status
    values = {"Content-Type": "text/html; charset=utf-8", **headers}
    reply.getheader.side_effect = lambda key, default=None: values.get(key, default)
    reply.read.side_effect = lambda limit: body[:limit]
    return reply


class MockWebMixin:
    def setUp(self):
        super().setUp()
        dns = patch("knowledge.scraping.socket.getaddrinfo", return_value=PUBLIC)
        self.dns = dns.start()
        self.addCleanup(dns.stop)
        connect = patch("knowledge.scraping._connect")
        self.connect = connect.start()
        self.addCleanup(connect.stop)
        self.connection = self.connect.return_value
        self.connection.getresponse.return_value = response()


class FetchTests(MockWebMixin, SimpleTestCase):
    def test_html_and_plain_text(self):
        for mime in ("text/html", "text/plain", "application/xhtml+xml"):
            with self.subTest(mime=mime):
                self.connection.getresponse.return_value = response(**{"Content-Type": mime})
                self.assertEqual(fetch_url(URL), (HTML, URL))
        self.connection.close.assert_called()

    def test_invalid_urls_never_connect(self):
        for url in ("file:///etc/passwd", "ftp://example.org/a", "http:///a", "http://user:pw@example.org",
                    "http://example.org:bad", "http://example.org:0", "http://example.org/\nfoo",
                    "http://example.org\\@localhost", "http://[::1", "http://example.org/" + "x" * 500):
            with self.subTest(url=url), self.assertRaises(ScrapingError):
                fetch_url(url)
        self.connect.assert_not_called()

    def test_nonpublic_and_mixed_dns_blocked(self):
        for address in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.2",
                        "0.0.0.0", "224.0.0.1", "::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1",
                        "4000::1", "64:ff9b::a00:1"):
            self.dns.return_value = PUBLIC + [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]
            with self.subTest(address=address), self.assertRaisesMessage(ScrapingError, "must be public"):
                fetch_url(URL)
        self.connect.assert_not_called()

    def test_public_relative_redirect_and_final_attribution(self):
        self.connection.getresponse.side_effect = [response(status=302, Location="/new"), response()]
        self.assertEqual(fetch_url(URL), (HTML, "https://example.org/new"))
        self.assertEqual(self.dns.call_count, 2)
        self.assertEqual(self.connection.close.call_count, 2)

    def test_redirect_to_private_host_never_connects_to_it(self):
        self.connection.getresponse.return_value = response(status=302, Location="http://localhost/secret")
        self.dns.side_effect = [PUBLIC, [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]]
        with self.assertRaisesMessage(ScrapingError, "must be public"):
            fetch_url(URL)
        self.assertEqual(self.connect.call_count, 1)

    def test_redirect_invalid_scheme_missing_location_and_loop(self):
        for headers, message in (({"Location": "file:///etc/passwd"}, "HTTP/HTTPS"),
                                 ({}, "Location"), ({"Location": URL}, "Too many redirects")):
            self.connection.getresponse.return_value = response(status=302, **headers)
            with self.subTest(headers=headers), self.assertRaisesMessage(ScrapingError, message):
                fetch_url(URL)

    def test_http_and_content_failures(self):
        cases = [
            (response(status=404), "HTTP 404"),
            (response(**{"Content-Type": "application/pdf"}), "Unsupported content"),
            (response(**{"Content-Encoding": "gzip"}), "Compressed"),
            (response(**{"Content-Length": str(MAX_BYTES + 1)}), "size limit"),
            (response(b"x" * (MAX_BYTES + 1)), "size limit"),
            (response(**{"Content-Length": "bad"}), "Invalid Content-Length"),
            (response(b"short", **{"Content-Length": "100"}), "Incomplete"),
            (response(b"\xff", **{"Content-Type": "text/plain; charset=utf-8"}), "decode"),
            (response(**{"Content-Type": "text/plain; charset=unknown-encoding"}), "decode"),
        ]
        for reply, message in cases:
            self.connection.getresponse.return_value = reply
            with self.subTest(message=message), self.assertRaisesMessage(ScrapingError, message):
                fetch_url(URL)

    def test_network_timeout_and_dns_failures(self):
        self.connection.getresponse.side_effect = TimeoutError("timed out")
        with self.assertRaisesMessage(ScrapingError, "timeout"):
            fetch_url(URL)
        self.connection.close.assert_called_once()
        self.dns.side_effect = socket.gaierror("unknown host")
        with self.assertRaisesMessage(ScrapingError, "resolve"):
            fetch_url(URL)


class ConnectionTests(SimpleTestCase):
    @patch("knowledge.scraping.ssl.create_default_context")
    @patch("knowledge.scraping.socket.socket")
    def test_pinned_ip_with_original_tls_hostname(self, socket_factory, context):
        connection = _connect("https", "example.org", 443, PUBLIC[0])
        socket_factory.return_value.connect.assert_called_once_with(("93.184.216.34", 443))
        context.return_value.wrap_socket.assert_called_once_with(socket_factory.return_value, server_hostname="example.org")
        self.assertEqual(connection.host, "example.org")
        self.assertIs(connection.sock, context.return_value.wrap_socket.return_value)
        connection.close()


@override_settings(RELAY_EMBEDDING_API_KEY="")
class IngestURLTests(MockWebMixin, TestCase):
    def setUp(self):
        super().setUp()
        retrieval.invalidate()
        self.addCleanup(retrieval.invalidate)

    def run_command(self, **kwargs):
        out = io.StringIO()
        call_command("ingest_url", URL, slug="scraped-appeals", title="FEMA Appeals", stdout=out, **kwargs)
        return out.getvalue()

    def test_stores_cleaned_chunks_and_retrieves(self):
        self.assertEqual(retrieval.retrieve("FEMA appeals documents"), [])
        self.assertIn("CREATED", self.run_command(disaster_type="flood", phase="recovery"))
        document = Document.objects.get()
        self.assertEqual(document.raw_text, HTML)
        self.assertEqual(document.clean_text, TEXT)
        self.assertEqual(document.source_type, "scrape")
        self.assertEqual(document.source_url, URL)
        self.assertEqual((document.disaster_type, document.phase), ("flood", "recovery"))
        self.assertTrue(document.chunks.exists())
        results = retrieval.retrieve("FEMA appeals documents")
        self.assertEqual(results[0].document_slug, document.slug)
        self.assertIn("60 days", results[0].text)
        self.assertEqual(results[0].url, URL)

    def test_repeat_then_update_preserves_identity_and_inactive_state(self):
        self.run_command()
        document = Document.objects.get()
        ids = list(document.chunks.values_list("id", flat=True))
        self.assertIn("UNCHANGED", self.run_command())
        self.assertEqual(ids, list(document.chunks.values_list("id", flat=True)))
        document.is_active = False
        document.save()
        self.connection.getresponse.return_value = response(HTML.replace("60 days", "sixty days").encode())
        self.assertIn("UPDATED", self.run_command())
        document.refresh_from_db()
        self.assertFalse(document.is_active)
        self.assertIn("sixty days", document.clean_text)
        self.assertNotEqual(ids, list(document.chunks.values_list("id", flat=True)))
        self.assertEqual(Document.objects.count(), 1)

    def test_collision_does_not_overwrite_other_sources(self):
        for source_type, source_url in (("manifest", URL), ("scrape", "https://example.org/unrelated")):
            ingest_document(slug="scraped-appeals", title="Original", raw_text=TEXT,
                            source_type=source_type, source_url=source_url)
            with self.assertRaisesMessage(CommandError, "different source"):
                self.run_command()
            self.assertEqual(Document.objects.get().title, "Original")

    def test_updated_content_refreshes_existing_retrieval_index(self):
        self.run_command()
        self.assertTrue(retrieval.retrieve("FEMA appeals documents"))
        updated = ("Wildfire evacuation planning: prepare emergency supplies and establish two escape routes. "
                   "Follow local evacuation instructions and keep family contact details readily available.")
        self.connection.getresponse.return_value = response(updated.encode(), **{"Content-Type": "text/plain"})
        self.run_command()
        results = retrieval.retrieve("wildfire evacuation")
        self.assertEqual(results[0].document_slug, "scraped-appeals")
        self.assertIn("escape routes", results[0].text)
        self.assertNotIn("60 days", results[0].text)

    def test_final_redirect_url_is_stored(self):
        self.connection.getresponse.side_effect = [response(status=301, Location="/new"), response()]
        self.run_command()
        self.assertEqual(Document.objects.get().source_url, "https://example.org/new")

    def test_failures_leave_no_documents(self):
        for reply in (response(b"<nav>Only menu</nav>"), response(status=500),
                      response(**{"Content-Type": "application/pdf"})):
            self.connection.getresponse.return_value = reply
            with self.assertRaises(CommandError):
                self.run_command()
            self.assertFalse(Document.objects.exists())

    def test_bad_metadata_is_rejected_before_fetch(self):
        with self.assertRaises(CommandError):
            call_command("ingest_url", URL, slug="bad slug", title="Title")
        self.connect.assert_not_called()
