import os.path
from typing import Any

from mechanize import HTTPError
from urllib.error import URLError
import socket
from mechanize._response import response_seek_wrapper as Response
from calibre.utils.localization import lang_as_iso639_1

from ..lib.utils import traceback_error, request
from ..lib.exception import UnexpectedResult

from .languages import lang_directionality


load_translations()


class Base:
    name: str | None = None
    alias: str | None = None
    free = False

    lang_codes: dict[str, Any] = {}
    config: dict[str, Any] = {}
    endpoint: str | None = None
    method = 'POST'
    headers: dict[str, str] = {}
    stream = False
    need_api_key = True
    api_key_hint = _('API Keys')
    api_key_pattern = r'^[^\s]+$'
    api_key_errors = ['401']
    separator = '\n\n'
    placeholder = ('{{{{id_{}}}}}', r'({{\s*)+id\s*_\s*{}\s*(\s*}})+')
    using_tip = None

    concurrency_limit: int = 0
    request_interval: float = 0.0
    request_attempt: int = 3
    request_timeout: float = 10.0
    max_error_count: int = 10

    def __init__(self):
        self.source_lang: str | None = None
        self.target_lang: str | None = None
        self.proxy_uri: str | None = None
        self.search_paths = []

        self.merge_enabled = False
        self.api_keys: list = self.config.get('api_keys', [])[:]
        self.bad_api_keys = []
        self.api_key = self.get_api_key()

        concurrency_limit = self.config.get('concurrency_limit')
        if concurrency_limit is not None:
            self.concurrency_limit = int(concurrency_limit)
        request_interval = self.config.get('request_interval')
        if request_interval is not None:
            self.request_interval = request_interval
        request_attempt = self.config.get('request_attempt')
        if request_attempt is not None:
            self.request_attempt = int(request_attempt)
        request_timeout = self.config.get('request_timeout')
        if request_timeout is not None:
            self.request_timeout = request_timeout
        max_error_count = self.config.get('max_error_count')
        if max_error_count is not None:
            self.max_error_count = max_error_count

    @classmethod
    def load_lang_codes(cls, codes):
        if not ('source' in codes or 'target' in codes):
            codes = {'source': codes, 'target': codes}
        return codes

    @classmethod
    def get_lang_directionality(cls, lang_code):
        return lang_directionality.get(lang_code, 'auto')

    @classmethod
    def get_source_code(cls, lang):
        source_codes: dict = cls.lang_codes.get('source') or {}
        return 'auto' if lang == _('Auto detect') else source_codes.get(lang)

    @classmethod
    def get_target_code(cls, lang):
        target_codes: dict = cls.lang_codes.get('target') or {}
        return target_codes.get(lang)

    @classmethod
    def get_iso639_target_code(cls, lang):
        return lang_as_iso639_1(cls.get_target_code(lang))

    @classmethod
    def set_config(cls, config):
        cls.config = config

    @classmethod
    def api_key_error_message(cls):
        return _('A correct key format "{}" is required.') \
            .format(cls.api_key_hint)

    def get_api_key(self):
        if self.need_api_key and self.api_keys:
            return self.api_keys.pop(0)
        return None

    def swap_api_key(self):
        """Change the API key if the previous one cannot be used."""
        if self.api_key not in self.bad_api_keys:
            self.bad_api_keys.append(self.api_key)
            self.api_key = self.get_api_key()
            if self.api_key is not None:
                return True
        return False

    def need_swap_api_key(self, error_message):
        if self.need_api_key and len(self.api_keys) > 0 \
                and self.match_error(error_message):
            return True
        return False

    def match_error(self, error_message):
        for error in self.api_key_errors:
            if error in error_message:
                return True
        return False

    def set_search_paths(self, paths):
        self.search_paths = paths

    def get_external_program(self, name, paths=[]):
        for path in paths + self.search_paths:
            if not path.endswith('%s%s' % (os.path.sep, name)):
                path = os.path.join(path, name)
            if os.path.isfile(path):
                return path
        return None

    def set_merge_enabled(self, enable):
        self.merge_enabled = enable

    def set_source_lang(self, source_lang):
        self.source_lang = source_lang

    def set_target_lang(self, target_lang):
        self.target_lang = target_lang

    def get_target_lang(self):
        return self.target_lang

    def set_proxy(self, proxy=[]):
        if isinstance(proxy, list) and len(proxy) == 2:
            self.proxy_uri = '%s:%s' % tuple(proxy)
            if not self.proxy_uri.startswith('http'):
                self.proxy_uri = 'http://%s' % self.proxy_uri

    def set_concurrency_limit(self, limit):
        self.concurrency_limit = limit

    def set_request_attempt(self, limit):
        self.request_attempt = limit

    def set_request_interval(self, seconds):
        self.request_interval = seconds

    def set_request_timeout(self, seconds):
        self.request_timeout = seconds

    def _get_source_code(self):
        return self.get_source_code(self.source_lang)

    def _get_target_code(self):
        return self.get_target_code(self.target_lang)

    def _is_auto_lang(self):
        return self._get_source_code() == 'auto'

    def translate(self, text):
        try:
            response = request(
                url=self.get_endpoint(), data=self.get_body(text),
                headers=self.get_headers(), method=self.method,
                timeout=self.request_timeout, proxy_uri=self.proxy_uri,
                raw_object=self.stream)
            return self.get_result(response)
        except Exception as e:
            # Combine the error messages for investigation.
            error_message = traceback_error()
            # If we failed before getting a response, surface a connection/timeout error,
            # otherwise treat it as a parse error and include the raw response.
            if isinstance(e, HTTPError):
                # Include body from HTTP errors when available
                try:
                    error_message += '\n\n' + e.read().decode('utf-8')
                except Exception:
                    pass
            elif isinstance(e, (URLError, socket.timeout, TimeoutError)) and 'response' not in locals():
                # Network-level failure (e.g., timed out, DNS, connection refused)
                network_message = _('Request to translation service failed (timeout or connectivity issue). Details: {}')
                # Swap a valid API key if necessary.
                if self.need_swap_api_key(str(e)) and self.swap_api_key():
                    return self.translate(text)
                raise UnexpectedResult(network_message.format(str(e)))

            if not self.stream and 'response' in locals():
                # We did get a response but couldn't parse/handle it; include the raw data.
                try:
                    error_message += '\n\n' + (response if isinstance(response, str) else '')
                except Exception:
                    pass
            # Swap a valid API key if necessary.
            if self.need_swap_api_key(error_message) and self.swap_api_key():
                return self.translate(text)
            raise UnexpectedResult(
                _('Can not parse returned response. Raw data: {}')
                .format('\n\n' + error_message))

    def get_endpoint(self):
        return self.endpoint

    def get_headers(self):
        return self.headers

    def get_body(self, text):
        return text

    def get_result(self, response: Response | bytes | str):
        return response

    def get_usage(self):
        return None
