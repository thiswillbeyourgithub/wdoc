"""Preserve MiniMax reasoning history with the minimum supported ChatLiteLLM."""

from langchain_litellm import ChatLiteLLM


class MiniMaxChatLiteLLM(ChatLiteLLM):
    def _create_message_dicts(self, messages, stop):
        message_dicts, params = super()._create_message_dicts(messages, stop)
        for message, serialized in zip(messages, message_dicts):
            if serialized.get("role") != "assistant":
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                serialized["reasoning_content"] = reasoning
                # ChatLiteLLM 0.6.5 also presents reasoning as a thinking content
                # block. Replay it in the API field, not twice in the history.
                content = serialized.get("content")
                if isinstance(content, list):
                    serialized["content"] = [
                        block
                        for block in content
                        if not (
                            isinstance(block, dict)
                            and block.get("type") == "thinking"
                            and block.get("thinking") == reasoning
                        )
                    ] or ""
        return message_dicts, params
