"""Small public-web fetcher. Connections use validated IPs, including on redirects."""
import http.client
import ipaddress
import socket
import ssl
from email.message import Message
from urllib.parse import urljoin, urlsplit, urlunsplit


MAX_BYTES = 2_000_000
MAX_REDIRECTS = 5
TIMEOUT = 10
REDIRECTS = {301, 302, 303, 307, 308}


class ScrapingError(ValueError):
    """A URL cannot safely provide supported source text."""


def public_destination(url):
    """Normalize a URL and resolve it once; reject mixed public/private DNS answers."""
    try:
        if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) == 127 for c in url):
            raise ValueError
        parts = urlsplit(url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None or "\\" in url):
            raise ValueError
        host = parts.hostname.encode("idna").decode("ascii")
        port = parts.port if parts.port is not None else (443 if parts.scheme == "https" else 80)
        if not 1 <= port <= 65535 or "%" in host:
            raise ValueError
        authority = f"[{host}]" if ":" in host else host
        if parts.port is not None:
            authority += f":{port}"
        normalized = urlunsplit((parts.scheme, authority, parts.path or "/", parts.query, ""))
        normalized.encode("ascii")
        if len(normalized) > 500:
            raise ValueError
    except (ValueError, UnicodeError):
        raise ScrapingError("Use an HTTP/HTTPS URL without credentials, whitespace or fragments in the host (maximum 500 characters).") from None
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ScrapingError(f"Could not resolve source host: {exc}") from exc
    if not addresses:
        raise ScrapingError("Source host has no addresses.")
    for _, _, _, _, address in addresses:
        ip = ipaddress.ip_address(address[0])
        if (not ip.is_global or ip.is_multicast or ip.is_reserved
                or any(getattr(ip, name, None) is not None for name in ("ipv4_mapped", "sixtofour", "teredo"))):
            raise ScrapingError("Source destination must be public; local/private/reserved addresses are blocked.")
    return normalized, parts.scheme, host, port, addresses[0]


def _connect(scheme, host, port, address):
    # Do not resolve again in HTTPConnection: pin the checked address to prevent
    # DNS rebinding. HTTPS still verifies certificates against the original host.
    family, socktype, proto, _, sockaddr = address
    sock = socket.socket(family, socktype, proto)
    try:
        sock.settimeout(TIMEOUT)
        sock.connect(sockaddr)
        if scheme == "https":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        connection = http.client.HTTPConnection(host, port, timeout=TIMEOUT)
        connection.sock = sock
        return connection
    except Exception:
        sock.close()
        raise


def fetch_url(url):
    """Return (decoded text, final URL); no proxies, cookies or automatic redirects."""
    try:
        for hop in range(MAX_REDIRECTS + 1):
            current, scheme, host, port, address = public_destination(url)
            connection = _connect(scheme, host, port, address)
            try:
                parts = urlsplit(current)
                target = urlunsplit(("", "", parts.path, parts.query, ""))
                connection.request("GET", target, headers={
                    "User-Agent": "Relay/1.0 (knowledge ingestion)",
                    "Accept": "text/html, text/plain, application/xhtml+xml",
                    "Accept-Encoding": "identity",
                })
                response = connection.getresponse()
                if response.status in REDIRECTS:
                    location = response.getheader("Location")
                    if not location:
                        raise ScrapingError("Redirect is missing its Location header.")
                    if hop == MAX_REDIRECTS:
                        raise ScrapingError("Too many redirects (maximum 5).")
                    url = urljoin(current, location)
                    continue
                if response.status != 200:
                    raise ScrapingError(f"Source returned HTTP {response.status}.")
                content_type = Message()
                content_type["content-type"] = response.getheader("Content-Type", "application/octet-stream")
                if content_type.get_content_type() not in {"text/html", "text/plain", "application/xhtml+xml"}:
                    raise ScrapingError("Unsupported content type; use HTML or plain text.")
                if response.getheader("Content-Encoding", "identity").lower() != "identity":
                    raise ScrapingError("Compressed responses are not supported.")
                length = response.getheader("Content-Length")
                if length is not None:
                    try:
                        size = int(length)
                    except ValueError:
                        raise ScrapingError("Invalid Content-Length header.") from None
                    if size < 0 or size > MAX_BYTES:
                        raise ScrapingError(f"Response exceeds size limit ({MAX_BYTES} bytes) or has invalid length.")
                body = response.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise ScrapingError(f"Response exceeds size limit ({MAX_BYTES} bytes).")
                if length is not None and len(body) != size:
                    raise ScrapingError("Incomplete response body.")
                try:
                    return body.decode(content_type.get_content_charset() or "utf-8-sig"), current
                except (UnicodeError, LookupError):
                    raise ScrapingError("Cannot decode response using its declared charset (default UTF-8).") from None
            finally:
                connection.close()
    except (OSError, http.client.HTTPException) as exc:
        raise ScrapingError(f"Source fetch failed (network/TLS/timeout): {exc}") from exc
