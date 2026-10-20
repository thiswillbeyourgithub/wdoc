"""Add model metadata that is not yet available in LiteLLM."""

from typing import Any, Literal

from loguru import logger


MINIMAX_PROVIDER = "minimax"

# Endpoint sources:
# https://platform.minimax.io/docs
# https://platform.minimaxi.com/docs
MINIMAX_ENDPOINTS = {
    "global_en": {
        "openai_base_url": "https://api.minimax.io/v1",
        "anthropic_base_url": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "openai_base_url": "https://api.minimaxi.com/v1",
        "anthropic_base_url": "https://api.minimaxi.com/anthropic",
    },
}

# Model metadata sources:
# https://platform.minimax.io/docs/guides/models-intro
# https://platform.minimax.io/docs/api-reference/text-openai-api
# https://platform.minimax.io/docs/guides/pricing
MINIMAX_MODELS = {
    "minimax/MiniMax-M3.1-Flash": {
        "litellm_provider": MINIMAX_PROVIDER,
        "mode": "chat",
        "max_tokens": 1_000_000,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 524_288,
        "input_cost_per_token": 0.15 / 1_000_000,
        "output_cost_per_token": 0.60 / 1_000_000,
        "cache_read_input_token_cost": 0.015 / 1_000_000,
        "cache_creation_input_token_cost": None,
        "input_modalities": ["text", "image", "video"],
        "thinking": ["adaptive"],
        "supports_vision": True,
        "supports_reasoning": True,
        "supports_adaptive_thinking": True,
        "supports_function_calling": True,
    },
    "minimax/MiniMax-M3": {
        "litellm_provider": MINIMAX_PROVIDER,
        "mode": "chat",
        "max_tokens": 1_000_000,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 524_288,
        # Conservative long-context estimate; standard <=512k requests cost half.
        "input_cost_per_token": 0.6 / 1_000_000,
        "output_cost_per_token": 2.4 / 1_000_000,
        "cache_read_input_token_cost": 0.12 / 1_000_000,
        "cache_creation_input_token_cost": None,
        "input_modalities": ["text", "image", "video"],
        "thinking": ["adaptive", "disabled"],
        "supports_vision": True,
        "supports_reasoning": True,
        "supports_adaptive_thinking": True,
    },
}

# Keep fallback accounting for explicitly requested legacy models.
MINIMAX_LEGACY_MODELS = {
    "minimax/MiniMax-M2.7": {
        "litellm_provider": MINIMAX_PROVIDER,
        "mode": "chat",
        "max_tokens": 204_800,
        "max_input_tokens": 204_800,
        "input_cost_per_token": 0.3 / 1_000_000,
        "output_cost_per_token": 1.2 / 1_000_000,
        "cache_read_input_token_cost": 0.06 / 1_000_000,
        "cache_creation_input_token_cost": 0.375 / 1_000_000,
        "input_modalities": ["text"],
        "thinking": ["always_on"],
        "supports_reasoning": True,
    },
}


# Verified OpenRouter IDs. Its pricing and token limits remain live gateway data:
# https://openrouter.ai/api/v1/models
OPENROUTER_MINIMAX_MODELS = (
    "openrouter/minimax/minimax-m3.1-flash:free",
    "openrouter/minimax/minimax-m3",
)


def is_current_minimax_choice(model_id, metadata):
    """Filter chat suggestions only; never remove explicit-request metadata."""
    if metadata.get("mode", "chat") != "chat":
        return True
    if model_id.startswith("minimax/"):
        return model_id in MINIMAX_MODELS
    if model_id.startswith("openrouter/minimax/"):
        return model_id in OPENROUTER_MINIMAX_MODELS
    return True


def get_minimax_api_base(
    region: Literal["global_en", "cn_zh"],
    protocol: Literal["openai", "anthropic"] = "openai",
) -> str:
    """Return the configured MiniMax base URL for a region and protocol."""
    try:
        return MINIMAX_ENDPOINTS[region][f"{protocol}_base_url"]
    except KeyError as err:
        raise ValueError(
            f"Unsupported MiniMax endpoint selection: {region}/{protocol}"
        ) from err


def _add_extra_models_metadata(litellm: Any) -> None:
    models_to_add = {
        model_id: metadata
        for model_id, metadata in (MINIMAX_MODELS | MINIMAX_LEGACY_MODELS).items()
        if model_id not in litellm.model_cost
    }
    if models_to_add:
        litellm.register_model(models_to_add)

    # Curate ordinary chat choices, keeping metadata for explicit legacy IDs and
    # preserving any non-chat capabilities registered by LiteLLM.
    provider_models = litellm.models_by_provider.get(MINIMAX_PROVIDER, [])
    non_chat_models = [
        model_id
        for model_id in provider_models
        if litellm.model_cost.get(
            model_id, litellm.model_cost.get(f"minimax/{model_id}", {})
        ).get("mode", "chat")
        != "chat"
    ]
    litellm.models_by_provider[MINIMAX_PROVIDER] = list(MINIMAX_MODELS) + [
        model_id for model_id in non_chat_models if model_id not in MINIMAX_MODELS
    ]


def add_extra_models_metadata(litellm: Any) -> None:
    """Add missing model metadata without preventing wdoc from starting."""
    try:
        _add_extra_models_metadata(litellm)
    except Exception as err:
        logger.warning(f"Could not add extra LiteLLM model metadata: {err}")
