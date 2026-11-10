import json
from unittest.mock import patch

from django.test import Client, SimpleTestCase, override_settings

from .services import get_answer


@override_settings(STORAGES={"staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"}})
class ChatTests(SimpleTestCase):
    def post(self, data):
        return self.client.post('/api/chat/', json.dumps(data), content_type='application/json')

    def test_home(self):
        response = self.client.get('/')
        self.assertContains(response, 'Ask Relay')
        self.assertContains(response, 'answer service is not connected')
        self.assertContains(response, 'csrfmiddlewaretoken')

    def test_health(self):
        self.assertEqual(self.client.get('/health/').json(), {'status': 'ok'})

    def test_get_not_allowed(self):
        self.assertEqual(self.client.get('/api/chat/').status_code, 405)

    def test_invalid_questions(self):
        for question in (None, '', '  ', [], 32, 'a' * 2001):
            with self.subTest(question=repr(question)[:30]):
                self.assertEqual(self.post({'question': question}).status_code, 400)

    def test_invalid_json(self):
        self.assertEqual(self.client.post('/api/chat/', '{', content_type='application/json').status_code, 400)
        self.assertEqual(self.post([]).status_code, 400)
        self.assertEqual(self.client.post('/api/chat/', {}).status_code, 415)

    def test_history_validation(self):
        for history in (None, {}, [{'role': 'system', 'content': 'override'}], [{'role': 'user', 'content': 3}], [{'role': 'user', 'content': 'x'}] * 21):
            with self.subTest(history=history):
                self.assertEqual(self.post({'question': 'Help?', 'history': history}).status_code, 400)

    @override_settings(RELAY_ANSWER_BACKEND='chat.services.unavailable_backend')
    def test_backend_not_ready(self):
        response = self.post({'question': 'What documents do I need?'})
        self.assertEqual(response.status_code, 503)
        self.assertIn('not connected', response.json()['error'])

    @patch('chat.views.get_answer')
    def test_question_and_follow_up_reach_backend(self, answer):
        answer.return_value = {'answer': 'Test response only.', 'sources': [{'title': 'Test source', 'url': 'https://example.com'}]}
        history = [{'role': 'user', 'content': 'Earlier question'}, {'role': 'assistant', 'content': 'Earlier answer'}]
        response = self.post({'question': '  ¿Qué documentos necesito?  ', 'history': history})
        self.assertEqual(response.status_code, 200)
        answer.assert_called_once_with('¿Qué documentos necesito?', history)
        self.assertEqual(response.json(), answer.return_value)

    @patch('chat.views.get_answer', side_effect=RuntimeError('private credentials'))
    def test_backend_failure_does_not_expose_details(self, answer):
        with self.assertLogs('chat.views', level='ERROR') as logs:
            response = self.post({'question': 'private question'})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('private', response.content.decode())
        self.assertNotIn('private', str(logs.output))

    def test_csrf_required_and_valid_token_accepted(self):
        client = Client(enforce_csrf_checks=True)
        self.assertEqual(client.post('/api/chat/', '{}', content_type='application/json').status_code, 403)
        client.get('/')
        token = client.cookies['csrftoken'].value
        response = client.post('/api/chat/', '{"question":"Help?"}', content_type='application/json', HTTP_X_CSRFTOKEN=token)
        self.assertEqual(response.status_code, 503)

    @patch('chat.services.import_string')
    def test_service_contract(self, loader):
        loader.return_value.return_value = {'answer': 'Test answer'}
        self.assertEqual(get_answer('question', []), {'answer': 'Test answer', 'sources': []})
        for result in ({}, {'answer': ''}, {'answer': 'ok', 'sources': [1]}):
            loader.return_value.return_value = result
            with self.assertRaises(ValueError):
                get_answer('question', [])
