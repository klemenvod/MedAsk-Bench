from config import (
    LLAMA_URL,
    MIXTRAL_URL,
    GPT_TEMPERATURE,
    LLAMA_MAX_TOKENS,
    MIXTRAL_MAX_TOKENS,
    CLAUDE_MAX_TOKENS,
    CLAUDE_TEMPERATURE,
    GEMINI_MAX_TOKENS,
    GEMINI_TEMPERATURE,
)

MODELS = {
    "gpt4": {
        "url": "gpt-4-turbo-preview",
        "temperature": GPT_TEMPERATURE,
    },
    "gpt3.5": {
        "url": "gpt-3.5-turbo",
        "temperature": GPT_TEMPERATURE,
    },
    "gpt4o": {
        "url": "gpt-4o",
        "temperature": GPT_TEMPERATURE,
    },
    "llama-3-70b-instruct": {
        "url": LLAMA_URL,
        "max_tokens": LLAMA_MAX_TOKENS,
    },
    "mixtral-8x7b": {
        "url": MIXTRAL_URL,
        "max_tokens": MIXTRAL_MAX_TOKENS,
    },
    "claude-opus": {
        "url": "claude-3-opus-20240229",
        "max_tokens": CLAUDE_MAX_TOKENS,
        "temperature": CLAUDE_TEMPERATURE,
    },
    "gemini": {
        "url": "gemini-1.5-pro",
        "max_tokens": GEMINI_MAX_TOKENS,
        "temperature": GEMINI_TEMPERATURE,
    },
}
