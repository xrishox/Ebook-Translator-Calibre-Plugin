from urllib.parse import urlsplit
import json

from .openai import ChatgptTranslate
from ..lib.utils import request


load_translations()


class OpenRouterTranslate(ChatgptTranslate):
    name = 'OpenRouter'
    alias = 'OpenRouter (OpenAI-compatible)'
    # Default OpenRouter chat completions endpoint
    endpoint = 'https://openrouter.ai/api/v1/chat/completions'
    # Provide a helpful hint for OpenRouter API keys
    api_key_hint = 'sk-or-...'

    def get_models(self):
        # OpenRouter models endpoint is /api/v1/models (not /v1/models)
        domain = '://'.join(urlsplit(self.endpoint, 'https')[:2])
        model_endpoint = f'{domain}/api/v1/models'
        response = request(
            model_endpoint, headers=self.get_headers(),
            proxy_uri=self.proxy_uri)
        data = json.loads(response)
        # OpenRouter returns a list under 'data' as well
        return [item['id'] for item in data.get('data', [])]


class ChatgptTranslate2(ChatgptTranslate):
    name = 'ChatGPT(2)'
    alias = 'ChatGPT (Alt 2)'
    # inherits default endpoint; user can override per-engine
    pass


class ChatgptTranslate3(ChatgptTranslate):
    name = 'ChatGPT(3)'
    alias = 'ChatGPT (Alt 3)'
    pass


class ChatgptTranslate4(ChatgptTranslate):
    name = 'ChatGPT(4)'
    alias = 'ChatGPT (Alt 4)'
    pass

