"""Exercise the real loader and SDK serialization without provider credentials."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage


@pytest.fixture
def loader(monkeypatch):
    # Import the real loader without starting document-loading integrations in
    # wdoc.__init__. This keeps SDK tests independent of optional ML packages.
    root = Path(__file__).parents[1] / "wdoc"
    for name, path in (
        ("wdoc", root),
        ("wdoc.utils", root / "utils"),
        ("wdoc.utils.tasks", root / "utils" / "tasks"),
        ("wdoc.utils.customs", root / "utils" / "customs"),
    ):
        spec = importlib.util.spec_from_file_location(
            name, path / "__init__.py", submodule_search_locations=[str(path)]
        )
        monkeypatch.setitem(sys.modules, name, importlib.util.module_from_spec(spec))
    import litellm

    metadata = importlib.import_module(
        "wdoc.utils.customs.add_extra_litellm_models_metadata"
    )
    metadata.add_extra_models_metadata(litellm)
    module = importlib.import_module("wdoc.utils.llm")
    monkeypatch.setenv("MINIMAX_API_KEY", "offline-fixture")
    monkeypatch.setenv("WDOC_PRIVATE_MODE", "false")
    monkeypatch.setenv("WDOC_NO_MODELNAME_MATCHING", "false")
    misc = importlib.import_module("wdoc.utils.misc")
    misc.model_name_matcher.cache_clear()
    yield module, misc, metadata
    misc.model_name_matcher.cache_clear()


@pytest.fixture
def requests(monkeypatch):
    captured = []

    def send(client, request, **kwargs):
        assert request.method == "POST"
        payload = json.loads(request.content)
        captured.append((str(request.url), payload))
        message = {
            "role": "assistant",
            "content": "<think>Keep this history.</think>Answer",
            "reasoning_content": "Preserve this reasoning.",
            "tool_calls": [
                {
                    "id": "call_lookup",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key":"value"}'},
                }
            ],
        }
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "offline-response",
                "object": "chat.completion",
                "created": 1,
                "model": payload["model"],
                "choices": [
                    {"index": 0, "message": message, "finish_reason": "tool_calls"}
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
        )

    async def async_send(client, request, **kwargs):
        return send(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    return captured


def make_llm(loader, model="minimax/", api_base=None, **kwargs):
    module, misc, _ = loader
    selected = misc.model_name_matcher(model)
    return module.load_llm(
        misc.ModelName(selected),
        llm_verbosity=False,
        llm_cache=False,
        api_base=api_base,
        private=False,
        tags=[],
        **kwargs,
    )


@pytest.mark.parametrize("region", ["global_en", "cn_zh"])
@pytest.mark.parametrize("effort", [None, "low", "medium", "high", "xhigh", "max"])
def test_flash_default_and_effort_at_http_boundary(loader, requests, region, effort):
    base = loader[2].get_minimax_api_base(region)
    kwargs = {} if effort is None else {"reasoning_effort": effort}
    llm = make_llm(loader, api_base=base, **kwargs)
    llm.invoke("Hello")
    url, payload = requests[-1]
    assert url == f"{base}/chat/completions"
    assert payload["model"] == "MiniMax-M3.1-Flash"
    assert payload["max_tokens"] == 524_288
    assert payload.get("reasoning_effort") == effort
    assert "service_tier" not in payload
    assert payload.get("thinking", {"type": "adaptive"}) == {"type": "adaptive"}
    assert payload.get("reasoning_split") is not False


@pytest.mark.parametrize(
    "model", ["minimax/minimax-m3.1-flash:free", "minimax/minimax-m3"]
)
def test_gateway_uses_its_own_contract(loader, requests, monkeypatch, model):
    import requests as requests_sdk
    from types import SimpleNamespace

    misc = loader[1]
    misc.get_openrouter_metadata.cache_clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    monkeypatch.setattr(
        requests_sdk,
        "get",
        lambda url: SimpleNamespace(
            json=lambda: {
                "data": [
                    {
                        "id": model,
                        "context_length": 300_000,
                        "top_provider": {"max_completion_tokens": 81_920},
                        "pricing": {
                            "prompt": "0",
                            "completion": "0",
                            "overrides": [{"prompt": "0"}],
                        },
                    }
                ]
            }
        ),
    )
    selected = "openrouter/minimax/" if "flash" in model else f"openrouter/{model}"
    llm = make_llm(loader, selected, reasoning_effort="high")
    llm.invoke("Hello")
    url, payload = requests[-1]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert payload["model"] == model
    assert payload["max_tokens"] == 81_920
    assert payload["reasoning"]["effort"] == "high"
    assert misc.get_openrouter_metadata()[f"openrouter/{model}"]["pricing"][
        "overrides"
    ] == [{"prompt": "0"}]
    misc.get_openrouter_metadata.cache_clear()


def test_gateway_saved_model_is_preserved(loader):
    assert (
        loader[1].model_name_matcher("openrouter/minimax/minimax-m2.7")
        == "openrouter/minimax/minimax-m2.7"
    )


@pytest.mark.parametrize("nested", [False, True])
def test_explicit_adaptive_thinking_and_nested_options(loader, requests, nested):
    options = {
        "thinking": {"type": "adaptive"},
        "reasoning_split": True,
        "reasoning_effort": "xhigh",
    }
    llm = make_llm(loader, **({"model_kwargs": options} if nested else options))
    llm.invoke("Hello")
    payload = requests[-1][1]
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["reasoning_split"] is True
    assert payload["reasoning_effort"] == "xhigh"


def test_async_flash_http_path(loader, requests):
    import asyncio

    asyncio.run(make_llm(loader, reasoning_effort="max").ainvoke("Hello"))
    url, payload = requests[-1]
    assert url == "https://api.minimax.io/v1/chat/completions"
    assert payload["model"] == "MiniMax-M3.1-Flash"
    assert payload["reasoning_effort"] == "max"


def test_m3_default_output_limit(loader, requests):
    make_llm(loader, "minimax/MiniMax-M3", thinking={"type": "disabled"}).invoke(
        "Hello"
    )
    assert requests[-1][1]["thinking"] == {"type": "disabled"}
    assert requests[-1][1]["max_tokens"] == 524_288
    assert requests[-1][1]["model"] == "MiniMax-M3"


def test_non_minimax_http_request_is_unchanged(loader, requests, monkeypatch):
    from langchain_litellm import ChatLiteLLM

    monkeypatch.setenv("WDOC_NO_MODELNAME_MATCHING", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    llm = make_llm(loader, "openai/gpt-4o-mini", max_tokens=321)
    assert type(llm) is ChatLiteLLM
    llm.invoke("Hello")
    url, payload = requests[-1]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert payload["model"] == "gpt-4o-mini"
    assert payload.get("max_tokens", payload.get("max_completion_tokens")) == 321


def test_default_endpoint(loader, requests):
    make_llm(loader).invoke("Hello")
    assert requests[-1][0] == "https://api.minimax.io/v1/chat/completions"


@pytest.mark.parametrize("model", ["MiniMax-M3", "MiniMax-M2.7", "custom-saved-model"])
def test_explicit_saved_models_and_custom_limits(loader, requests, model):
    import litellm

    model_id = f"minimax/{model}"
    if model == "custom-saved-model":
        litellm.register_model(
            {
                model_id: {
                    "max_tokens": 8192,
                    "litellm_provider": "minimax",
                    "mode": "chat",
                }
            }
        )
    llm = make_llm(
        loader, model_id, api_base="https://custom.invalid/v1", max_tokens=1234
    )
    llm.invoke("Hello")
    url, payload = requests[-1]
    assert url == "https://custom.invalid/v1/chat/completions"
    assert payload["model"] == model
    assert payload["max_tokens"] == 1234


def test_tool_reasoning_history_and_multimodal_input(loader, requests):
    llm = make_llm(loader, reasoning_effort="high")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                },
            },
        }
    ]
    user = HumanMessage(
        content=[
            {"type": "text", "text": "Describe these."},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.invalid/image.png"},
            },
            {"type": "video_url", "video_url": {"url": "mm_file://fixture"}},
        ]
    )
    response = llm.invoke([user], tools=tools)
    assert response.additional_kwargs["reasoning_content"] == "Preserve this reasoning."
    llm.invoke(
        [user, response, ToolMessage(content="result", tool_call_id="call_lookup")],
        tools=tools,
    )
    payload = requests[-1][1]
    assert payload["tools"] == tools
    assert payload["messages"][0]["content"] == user.content
    assert payload["messages"][1]["reasoning_content"] == "Preserve this reasoning."
    assert payload["messages"][1]["content"] == [
        {"type": "text", "text": "<think>Keep this history.</think>Answer"}
    ]
    assert payload["messages"][1]["tool_calls"][0]["id"] == "call_lookup"
    assert payload["messages"][2]["tool_call_id"] == "call_lookup"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning_effort": "none"},
        {"reasoning_effort": "minimal"},
        {"thinking": {"type": "disabled"}},
        {"thinking": {"type": "enabled", "budget_tokens": 100}},
        {"service_tier": "flex"},
        {"reasoning_split": False},
    ],
)
@pytest.mark.parametrize("location", ["top_level", "model_kwargs", "extra_body"])
def test_unsupported_flash_options_fail_before_http(loader, requests, kwargs, location):
    if location != "top_level":
        kwargs = {location: kwargs}
    with pytest.raises(ValueError, match="Flash"):
        make_llm(loader, **kwargs)
    assert requests == []


def test_non_minimax_selection_is_unchanged(loader, monkeypatch):
    monkeypatch.setenv("WDOC_NO_MODELNAME_MATCHING", "true")
    misc = loader[1]
    assert misc.model_name_matcher("other/custom") == "other/custom"
    assert misc.env.WDOC_DEFAULT_MODEL == "openrouter/deepseek/deepseek-v4-pro"
