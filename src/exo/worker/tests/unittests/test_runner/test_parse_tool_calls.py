"""Tests for parse_tool_calls generator, especially unclosed tool call handling."""

import json
from collections.abc import Generator
from typing import Any

from mlx_lm.tool_parsers import qwen3_coder

from exo.api.types import FinishReason
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.runner.llm_inference.model_output_parsers import parse_tool_calls
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser


def _make_responses(
    texts: list[str], finish_reason: FinishReason = "stop"
) -> Generator[GenerationResponse]:
    """Create a sequence of GenerationResponses from text strings."""
    for i, text in enumerate(texts):
        is_last = i == len(texts) - 1
        yield GenerationResponse(
            text=text,
            token=i,
            finish_reason=finish_reason if is_last else None,
            usage=None,
        )


def _dummier_parser(text: str) -> dict[str, Any]:
    return {"name": "test_fn", "arguments": {"arg": text}}


_dummy_parser = make_mlx_parser("<tool_call>", "</tool_call>", _dummier_parser)


class TestParseToolCalls:
    """Tests for parse_tool_calls generator."""

    def test_closed_tool_call_works_normally(self):
        """Normal tool call flow should not be affected."""
        texts = ["<tool_call>", "test_fn", "</tool_call>"]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                _dummy_parser,
                tools=None,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], ToolCallResponse)

    def test_no_tool_call_passes_through(self):
        """Responses without tool calls should pass through unchanged."""
        texts = ["Hello", " world"]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                _dummy_parser,
                tools=None,
            )
        )

        assert len(results) == 2
        assert all(isinstance(r, GenerationResponse) for r in results)
        r0 = results[0]
        r1 = results[1]
        assert isinstance(r0, GenerationResponse)
        assert isinstance(r1, GenerationResponse)
        assert r0.text == "Hello"
        assert r1.text == " world"
        assert r1.finish_reason == "stop"

    def test_failed_parse_yields_text(self):
        """A complete but unparseable tool call is model output, not a server
        failure: the raw text is yielded as-is with the model's real
        finish_reason rather than being relabeled "error" (which downstream
        serializes as an InternalServerError chunk)."""

        def _failing_parser(text: str) -> dict[str, Any]:
            raise ValueError("parse failed")

        texts = ["<tool_call>", "bad content", "</tool_call>"]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                make_mlx_parser("<tool_call>", "</tool_call>", _failing_parser),
                tools=None,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "<tool_call>bad content</tool_call>"
        assert results[0].finish_reason == "stop"

    def test_failed_parse_mid_stream_keeps_streaming(self):
        """An unparseable tool call in the middle of a turn must not end the
        stream: the tokens after it, including the final finish_reason, still
        reach the client."""

        def _failing_parser(text: str) -> dict[str, Any]:
            raise ValueError("parse failed")

        texts = ["<tool_call>", "bad content", "</tool_call>", " and then", " more"]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                make_mlx_parser("<tool_call>", "</tool_call>", _failing_parser),
                tools=None,
            )
        )

        assert len(results) == 3
        assert all(isinstance(r, GenerationResponse) for r in results)
        r0, r1, r2 = results
        assert isinstance(r0, GenerationResponse)
        assert isinstance(r1, GenerationResponse)
        assert isinstance(r2, GenerationResponse)
        assert r0.text == "<tool_call>bad content</tool_call>"
        assert r0.finish_reason is None
        assert r1.text == " and then"
        assert r2.text == " more"
        assert r2.finish_reason == "stop"

    def test_failed_parse_after_successful_call_still_emits_tool_calls(self):
        """If an earlier tool call in the turn parsed fine and a later one does
        not, the good calls are still emitted (as the terminal ToolCallResponse)
        and the bad one is surfaced as text without a finish_reason of its own."""

        def _picky_parser(text: str) -> dict[str, Any]:
            if "good" not in text:
                raise ValueError("parse failed")
            return {"name": "test_fn", "arguments": {"arg": text.strip()}}

        texts = [
            "<tool_call>",
            "good",
            "</tool_call>",
            "<tool_call>",
            "bad",
            "</tool_call>",
        ]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                make_mlx_parser("<tool_call>", "</tool_call>", _picky_parser),
                tools=None,
            )
        )

        assert len(results) == 2
        text_response, tool_response = results
        assert isinstance(text_response, GenerationResponse)
        assert text_response.text == "<tool_call>bad</tool_call>"
        assert text_response.finish_reason is None
        assert isinstance(tool_response, ToolCallResponse)
        assert len(tool_response.tool_calls) == 1
        assert tool_response.tool_calls[0].name == "test_fn"

    def test_truncated_tool_call_preserves_finish_reason(self):
        """A tool call cut off by the model's stop reason (no closing tag) is a
        normal truncated completion, not an error: the partial text is yielded
        with the model's real finish_reason rather than being relabeled "error"
        (which downstream serializes as an InternalServerError chunk)."""
        texts = ["<tool_call>", "partial cont"]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                _dummy_parser,
                tools=None,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "<tool_call>partial cont"
        assert results[0].finish_reason == "stop"

    def test_tool_call_truncated_by_length_preserves_length(self):
        """The observed production failure: a Qwen3.6 `write` call whose
        `content` argument is a whole HTML file runs out of max_tokens before
        `</tool_call>`. The client must see finish_reason "length" and the
        partial text, not an error chunk."""
        texts = [
            "<tool_call>",
            "\n<function=write>\n<parameter=file_path>\n/tmp/pipeline.html\n",
            "</parameter>\n<parameter=content>\n<!DOCTYPE html>\n<html>\n",
            "<body><p>cut off here",
        ]
        results = list(
            parse_tool_calls(
                _make_responses(texts, finish_reason="length"),
                make_mlx_parser(
                    "<tool_call>", "</tool_call>", qwen3_coder.parse_tool_call
                ),
                tools=None,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], GenerationResponse)
        assert results[0].text == "".join(texts)
        assert results[0].finish_reason == "length"

    def test_qwen3_coder_xml_with_multiline_html_content_parses(self):
        """The Hermes/XML form Qwen3.6 emits is what mlx-lm's qwen3_coder parser
        expects; a `content` parameter holding a multi-line HTML document with
        angle brackets and quotes must round-trip as a string argument."""
        html = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '    <meta charset="UTF-8">\n'
            "    <style>\n"
            "        body { color: white; }\n"
            "    </style>\n"
            "</head>\n"
            "<body>\n"
            '    <svg><path d="M95,140 C100,132 160,128 235,140" fill="none"/></svg>\n'
            "    <p>a < b && c > d</p>\n"
            "</body>\n"
            "</html>"
        )
        texts = [
            "<tool_call>",
            "\n<function=write>\n<parameter=file_path>\n/tmp/pipeline.html\n</parameter>\n",
            f"<parameter=content>\n{html}\n</parameter>\n",
            "<parameter=overwrite>\ntrue\n</parameter>\n</function>\n",
            "</tool_call>",
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                            "overwrite": {"type": "boolean"},
                        },
                        "required": ["file_path", "content"],
                    },
                },
            }
        ]
        results = list(
            parse_tool_calls(
                _make_responses(texts),
                make_mlx_parser(
                    "<tool_call>", "</tool_call>", qwen3_coder.parse_tool_call
                ),
                tools,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], ToolCallResponse)
        assert len(results[0].tool_calls) == 1
        call = results[0].tool_calls[0]
        assert call.name == "write"
        args = json.loads(call.arguments)  # pyright: ignore[reportAny]
        assert args == {
            "file_path": "/tmp/pipeline.html",
            "content": html,
            "overwrite": True,
        }

    def test_tool_schema_coerces_string_arguments_to_expected_types(self):
        """Tool argument values should be coerced using provided JSON schema."""

        def _parser_with_string_args(_text: str) -> dict[str, Any]:
            return {
                "name": "process",
                "arguments": {
                    "action": "output",
                    "id": "0",
                    "verbose": "true",
                    "temperature": "0.75",
                },
            }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process",
                    "description": "Manage background processes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "id": {"type": "integer"},
                            "verbose": {"type": "boolean"},
                            "temperature": {"type": "number"},
                        },
                        "required": ["action"],
                    },
                },
            }
        ]

        results = list(
            parse_tool_calls(
                _make_responses(["<tool_call>", "process", "</tool_call>"]),
                make_mlx_parser(
                    "<tool_call>", "</tool_call>", _parser_with_string_args
                ),
                tools,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], ToolCallResponse)

        args = json.loads(results[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        assert args == {
            "action": "output",
            "id": 0,
            "verbose": True,
            "temperature": 0.75,
        }

    def test_schema_coercion_skips_unknown_tools(self):
        """If no matching tool schema exists, arguments should remain unchanged."""

        def _parser_with_string_id(_text: str) -> dict[str, Any]:
            return {
                "name": "process",
                "arguments": {"action": "output", "id": "0"},
            }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "different_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                    },
                },
            }
        ]

        results = list(
            parse_tool_calls(
                _make_responses(["<tool_call>", "process", "</tool_call>"]),
                make_mlx_parser("<tool_call>", "</tool_call>", _parser_with_string_id),
                tools,
            )
        )

        assert len(results) == 1
        assert isinstance(results[0], ToolCallResponse)

        args = json.loads(results[0].tool_calls[0].arguments)  # pyright: ignore[reportAny]
        assert args == {"action": "output", "id": "0"}
