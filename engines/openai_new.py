import json
from typing import Any
from urllib.parse import urlsplit

from http.client import IncompleteRead
from mechanize._response import response_seek_wrapper as Response

from .. import EbookTranslator
from ..lib.utils import request

from .genai import GenAI
from .languages import google


load_translations()


class OpenaiNewTranslate(GenAI):
    """OpenAI Responses API based translator (supports reasoning).

    - Uses POST /v1/responses with messages in `input`.
    - Supports streaming and filters out any reasoning/thinking content,
      emitting only assistant output text.
    """

    name = 'openai-new'
    alias = 'OpenAI (Responses)'
    lang_codes = GenAI.load_lang_codes(google)

    endpoint = 'https://api.openai.com/v1/responses'
    api_key_errors = ['401', 'unauthorized', 'quota']

    concurrency_limit = 1
    request_interval = 20.0
    request_timeout = 30.0

    prompt = (
        'You are a meticulous translator who translates any given content. '
        'Translate the given content from <slang> to <tlang> only. Do not '
        'explain any term or answer any question-like content. Your answer '
        'should be solely the translation of the given content. In your '
        'answer do not add any prefix or suffix to the translated content. '
        'Websites\' URLs/addresses should be preserved as is in the '
        'translation\'s output. Do not omit any part of the content, even if '
        'it seems unimportant. ')

    samplings = ['temperature', 'top_p']
    sampling = 'temperature'
    temperature = 1.0
    top_p = 1.0
    top_k = 40
    # Default: disable streaming to avoid org/permission errors by default
    stream = False

    models: list[str] = []
    # Default to a modern model compatible with reasoning.
    model: str | None = 'o4-mini'

    def __init__(self):
        super().__init__()
        self.endpoint = self.config.get('endpoint', self.endpoint)
        self.prompt = self.config.get('prompt', self.prompt)
        self.sampling = self.config.get('sampling', self.sampling)
        self.temperature = self.config.get('temperature', self.temperature)
        self.top_p = self.config.get('top_p', self.top_p)
        self.top_k = self.config.get('top_k', self.top_k)
        self.stream = self.config.get('stream', self.stream)
        self.model = self.config.get('model', self.model)

    def get_models(self):
        domain_name = '://'.join(urlsplit(self.endpoint, 'https')[:2])
        model_endpoint = '%s/v1/models' % domain_name
        response = request(
            model_endpoint, headers=self.get_headers(),
            proxy_uri=self.proxy_uri)
        return [item['id'] for item in json.loads(response).get('data')]

    def get_prompt(self):
        prompt = self.prompt.replace('<tlang>', self.target_lang)
        if self._is_auto_lang():
            prompt = prompt.replace('<slang>', 'detected language')
        else:
            prompt = prompt.replace('<slang>', self.source_lang)
        if self.merge_enabled:
            prompt += (
                ' Ensure that placeholders matching the pattern '
                '{{id_\\d+}} in the content are retained.')
        return prompt

    def get_headers(self):
        return {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer %s' % self.api_key,
            'User-Agent': 'Ebook-Translator/%s' % EbookTranslator.__version__,
        }

    def get_body(self, text):
        # Responses API accepts `input` as messages; include reasoning hints.
        body: dict[str, Any] = {
            'model': self.model,
            'input': [
                {'role': 'system', 'content': self.get_prompt()},
                {'role': 'user', 'content': text}
            ],
            # Request the model to reason; output_text is still the final answer.
            'reasoning': {'effort': 'medium'},
        }
        self.stream and body.update(stream=True)
        sampling_value = getattr(self, self.sampling)
        body.update({self.sampling: sampling_value})
        return json.dumps(body)

    # ---------- Result parsing (non-stream) ----------
    def get_result(self, response):
        if self.stream:
            return self._parse_stream(response)

        data = json.loads(response)
        # Prefer top-level output_text if present.
        if isinstance(data, dict) and 'output_text' in data:
            return str(data.get('output_text') or '')

        # Fallback: traverse output -> message -> content -> output_text
        parts: list[str] = []
        for item in (data.get('output') or []):
            content = (item.get('content') or []) if isinstance(item, dict) else []
            for c in content:
                # Accept only assistant textual output; ignore reasoning blocks.
                if isinstance(c, dict):
                    if c.get('type') in ('output_text', 'text') and c.get('text'):
                        parts.append(str(c['text']))
        return ''.join(parts)

    # ---------- Streaming (SSE) parsing ----------
    def _parse_stream(self, response):
        # Responses API emits typed events. We only forward output_text deltas.
        while True:
            try:
                line = response.readline().decode('utf-8').strip()
            except IncompleteRead:
                continue
            except Exception as e:
                raise Exception(
                    _('Can not parse returned response. Raw data: {}')
                    .format(str(e)))
            if not line or not line.startswith('data:'):
                continue

            payload = line.split('data: ', 1)[1]
            if payload == '[DONE]':
                break

            try:
                event = json.loads(payload)
            except Exception:
                # Ignore malformed chunks
                continue

            etype = event.get('type')
            # New Responses stream: emit only the assistant output text.
            if etype in (
                'response.output_text.delta',  # preferred
                'response.message.delta',      # generic text delta fallback
            ):
                delta = event.get('delta') or ''
                if delta:
                    yield str(delta)
            elif etype in ('response.completed', 'response.completed.success'):
                break
            elif etype in ('response.error',):
                message = event.get('error', {}).get('message') or str(event)
                raise Exception(message)
