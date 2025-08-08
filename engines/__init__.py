from .google import (
    GoogleFreeTranslateNew, GoogleFreeTranslateHtml, GoogleFreeTranslate,
    GoogleBasicTranslate, GoogleBasicTranslateADC, GoogleAdvancedTranslate,
    GeminiTranslate)
from .openai import ChatgptTranslate
from .openai_variants import (
    OpenRouterTranslate, ChatgptTranslate2, ChatgptTranslate3, ChatgptTranslate4)
from .openai_new import OpenaiNewTranslate
from .anthropic import ClaudeTranslate
from .deepl import DeeplTranslate, DeeplProTranslate, DeeplFreeTranslate
from .youdao import YoudaoTranslate
from .baidu import BaiduTranslate
from .microsoft import MicrosoftEdgeTranslate, AzureChatgptTranslate
from .deepseek import DeepseekTranslate

builtin_engines = (
    GoogleFreeTranslateNew, GoogleFreeTranslateHtml, GoogleFreeTranslate,
    GoogleBasicTranslate, GoogleBasicTranslateADC, GoogleAdvancedTranslate,
    ChatgptTranslate, ChatgptTranslate2, ChatgptTranslate3, ChatgptTranslate4,
    OpenRouterTranslate, OpenaiNewTranslate, AzureChatgptTranslate, GeminiTranslate, ClaudeTranslate,
    DeepseekTranslate, DeeplTranslate, DeeplProTranslate, DeeplFreeTranslate,
    MicrosoftEdgeTranslate, YoudaoTranslate, BaiduTranslate)
