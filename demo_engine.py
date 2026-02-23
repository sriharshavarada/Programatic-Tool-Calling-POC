"""Core demo engine: 3 execution styles with structured logs and metrics.

Modes:
- local_langgraph: Claude writes Python code, host executes locally (LangGraph orchestration)
- sandbox_langgraph: Claude managed code execution + local tool callbacks (LangGraph orchestration)
- traditional: no programmatic tool calling; host fetches data and asks Claude directly
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from anthropic_http import (
    AnthropicAPIError,
    extract_text_blocks,
    load_dotenv_if_present,
    post_messages,
    strip_code_fences,
)
from shared_tool import healthcare_claims_get_denials


DATES = ["2026-02-18", "2026-02-19", "2026-02-20"]
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
DEFAULT_CODE_EXEC_TOOL_TYPE = os.environ.get(
    "ANTHROPIC_CODE_EXEC_TOOL_TYPE", "code_execution_20260120"
)


@dataclass
class LogEvent:
    ts_ms: int
    level: str
    message: str


@dataclass
class DemoResult:
    mode: str
    ok: bool
    started_at_ms: int
    duration_ms: int
    logs: list[LogEvent] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_code: str | None = None
    final_text: str | None = None
    execution_stdout: str | None = None
    prompt_text: str | None = None
    prompt_preview: str | None = None
    integration_snippet: str | None = None
    llm_transcript: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["logs"] = [asdict(item) for item in self.logs]
        return data


class Logger:
    def __init__(self, result: DemoResult, prefix: str) -> None:
        self.result = result
        self.prefix = prefix

    def log(self, message: str, level: str = "info") -> None:
        self.result.logs.append(
            LogEvent(ts_ms=int(time.time() * 1000), level=level, message=message)
        )

    def capture_stdout_call(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            value = fn(*args, **kwargs)
        for line in buf.getvalue().splitlines():
            self.log(line)
        return value


def _langgraph_imports() -> tuple[Any, Any, Any]:
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - dependency/runtime specific
        try:
            from langgraph.graph import END, StateGraph  # type: ignore
            START = "__start__"
        except Exception:
            raise RuntimeError(
                "Missing langgraph. Install requirements first (see requirements.txt)."
            ) from exc
    return StateGraph, START, END


def _langchain_anthropic_import() -> Any:
    try:
        from langchain_anthropic import ChatAnthropic
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing langchain-anthropic. Install requirements first (see requirements.txt)."
        ) from exc
    return ChatAnthropic


def _extract_ai_text_from_langchain_message(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    return str(content)


def _extract_usage_from_langchain_message(message: Any) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    raw_usage = getattr(message, "usage_metadata", None)
    if isinstance(raw_usage, dict):
        usage.update(raw_usage)
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        rm_usage = response_metadata.get("usage")
        if isinstance(rm_usage, dict):
            usage.setdefault("response_metadata_usage", rm_usage)
        stop_reason = response_metadata.get("stop_reason")
        if stop_reason is not None:
            usage.setdefault("stop_reason", stop_reason)
    return usage


def _merge_usage(acc: dict[str, Any], usage: dict[str, Any], logger: Logger, label: str) -> None:
    if not usage:
        return
    acc.setdefault("calls", [])
    acc["calls"].append({"label": label, **usage})
    _recompute_usage_totals(acc)
    logger.log(f"{logger.prefix} token usage ({label}) = {usage}")


def _usage_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _recompute_usage_totals(acc: dict[str, Any]) -> None:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "api_calls": 0,
    }
    for call in acc.get("calls", []):
        if not isinstance(call, dict):
            continue
        totals["api_calls"] += 1
        src = call.get("raw_usage") if isinstance(call.get("raw_usage"), dict) else call
        if not isinstance(src, dict):
            continue
        totals["input_tokens"] += _usage_int(src.get("input_tokens"))
        totals["output_tokens"] += _usage_int(src.get("output_tokens"))
        totals["cache_creation_input_tokens"] += _usage_int(
            src.get("cache_creation_input_tokens")
        )
        totals["cache_read_input_tokens"] += _usage_int(src.get("cache_read_input_tokens"))
        totals["total_tokens"] += _usage_int(src.get("total_tokens"))
    if totals["total_tokens"] == 0:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    acc["totals"] = totals


def _safe_json_preview(obj: Any, limit: int = 500) -> str:
    text = json.dumps(obj, ensure_ascii=True, default=str)
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def _append_transcript(result: DemoResult, heading: str, content: Any) -> None:
    try:
        if isinstance(content, str):
            body = content
        else:
            body = json.dumps(content, indent=2, ensure_ascii=True, default=str)
    except Exception:
        body = str(content)
    chunk = f"{heading}\n{body}".strip()
    if result.llm_transcript:
        result.llm_transcript += "\n\n" + chunk
    else:
        result.llm_transcript = chunk


def _content_blocks_as_jsonable(response: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in response.get("content", []):
        if not isinstance(block, dict):
            blocks.append({"type": "unknown", "value": str(block)})
            continue
        b = dict(block)
        blocks.append(b)
    return blocks


def _local_prompt() -> tuple[str, str]:
    system_prompt = (
        "You generate runnable Python code only. Do not use markdown fences. "
        "Do not explain anything. No imports. Use only builtins and the provided "
        "function healthcare_claims_get_denials(service_date)."
    )
    user_prompt = """
You are a healthcare revenue-cycle analyst assistant.

Task:
- Analyze denied claims for these service dates:
  - 2026-02-18
  - 2026-02-19
  - 2026-02-20
- Use the available tool function healthcare_claims_get_denials(service_date)
- The tool returns rows shaped like:
  {"claim_id": str, "patient_id": str, "payer": str, "denial_reason": str, "denied_amount_NEW": int}
- Print observable logs labeled [LOCAL]
- Do not print every row; print counts and summaries only
- Produce a concise terminal summary that includes:
  - total denied amount across all dates
  - denied amount by denial_reason
  - denied amount by payer
  - the top denial reason by total denied amount

Assume healthcare_claims_get_denials(service_date: str) is already available in scope.
Use the exact field name denied_amount_NEW for amount calculations.
Decide the code structure yourself.
""".strip()
    return system_prompt, user_prompt


def _local_summary_prompt(execution_stdout: str) -> str:
    return (
        "You are reviewing the output of a healthcare revenue-cycle analysis program.\n"
        "Summarize the result in plain English with:\n"
        "- total denied amount\n"
        "- top denial reason\n"
        "- top payer by denied amount (if inferable)\n"
        "- one operational recommendation\n\n"
        "Program stdout:\n"
        f"{execution_stdout}"
    )


def _traditional_prompt_prefix() -> str:
    return (
        "You are a healthcare revenue-cycle analyst assistant. "
        "Use classic tool calling (no code execution / no programmatic tool calling) to analyze "
        "denied claims for service dates 2026-02-18, 2026-02-19, and 2026-02-20 using the tool "
        "healthcare_claims_get_denials(service_date). Compute total denied amount, denied amount "
        "by denial_reason, denied amount by payer, and top denial reason by denied amount. "
        "The amount field is denied_amount_NEW."
    )


def _local_integration_snippet(model: str) -> str:
    return f"""# Local (LangGraph): Claude generates Python, host executes locally
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="{model}", max_tokens=1200, temperature=0)
msg = llm.invoke([
    ("system", "You generate runnable Python code only..."),
    ("human", "Analyze denied claims for 2026-02-18/19/20 using healthcare_claims_get_denials(service_date)..."),
])
generated_code = msg.content if isinstance(msg.content, str) else str(msg.content)

exec_globals = {{
    "__builtins__": __builtins__,
    "healthcare_claims_get_denials": healthcare_claims_get_denials,
}}
exec(generated_code, exec_globals, {{}})
"""


def _sandbox_integration_snippet(model: str, code_exec_tool_type: str) -> str:
    return f"""# Sandbox (programmatic tool calling): Anthropic managed code_execution + local tool callback
tools = [
    {{"type": "{code_exec_tool_type}", "name": "code_execution"}},
    {{
        "name": "healthcare_claims_get_denials",
        "input_schema": {{
            "type": "object",
            "properties": {{"service_date": {{"type": "string"}}}},
            "required": ["service_date"]
        }},
        "allowed_callers": ["{code_exec_tool_type}"]
    }}
]

conversation = [{{"role": "user", "content": "Use code_execution to analyze denials for 2026-02-18/19/20..."}}]
while True:
    response = post_messages({{
        "model": "{model}",
        "max_tokens": 2200,
        "messages": conversation,
        "tools": tools,
    }}, log_prefix="[SANDBOX]")
    conversation.append({{"role": "assistant", "content": response["content"]}})

    tool_results = []
    for block in response.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "healthcare_claims_get_denials":
            rows = healthcare_claims_get_denials(block["input"]["service_date"])
            tool_results.append({{
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": json.dumps(rows),
            }})
    if not tool_results:
        break
    conversation.append({{"role": "user", "content": tool_results}})
"""


def _traditional_integration_snippet(model: str) -> str:
    return f"""# Traditional (classic tool calling): no code_execution, no generated Python
tools = [{{
    "name": "healthcare_claims_get_denials",
    "input_schema": {{
        "type": "object",
        "properties": {{"service_date": {{"type": "string"}}}},
        "required": ["service_date"]
    }}
}}]

conversation = [{{"role": "user", "content": "Use healthcare_claims_get_denials for 2026-02-18/19/20 and summarize..."}}]
while True:
    response = post_messages({{
        "model": "{model}",
        "max_tokens": 1200,
        "messages": conversation,
        "tools": tools,
    }}, log_prefix="[TRADITIONAL]")
    conversation.append({{"role": "assistant", "content": response["content"]}})

    tool_results = []
    for block in response.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "healthcare_claims_get_denials":
            rows = healthcare_claims_get_denials(block["input"]["service_date"])
            tool_results.append({{
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": json.dumps(rows),
            }})
    if not tool_results:
        break
    conversation.append({{"role": "user", "content": tool_results}})
"""


def _traditional_user_prompt() -> str:
    return (
        "You are a healthcare revenue-cycle analyst assistant.\n"
        "Use the available tool healthcare_claims_get_denials(service_date) to analyze denied claims for:\n"
        "- 2026-02-18\n"
        "- 2026-02-19\n"
        "- 2026-02-20\n\n"
        "This is a traditional tool-calling flow (no code execution tool, no generated Python).\n"
        "Call the tool as needed, then produce a concise report with:\n"
        "- total denied amount\n"
        "- denied amount by denial_reason\n"
        "- denied amount by payer\n"
        "- top denial reason by denied amount\n"
        "The amount field is denied_amount_NEW.\n"
        "Do not write code; use tool calls and then summarize."
    )


def run_local_langgraph(model: str | None = None) -> DemoResult:
    load_dotenv_if_present()
    model = model or DEFAULT_MODEL
    started = int(time.time() * 1000)
    result = DemoResult(
        mode="local_langgraph", ok=False, started_at_ms=started, duration_ms=0
    )
    logger = Logger(result, prefix="[LOCAL]")
    logger.log("[LOCAL] Mode B starting (LangGraph + local execution)")
    local_system_prompt, local_user_prompt = _local_prompt()
    result.prompt_text = f"[SYSTEM]\n{local_system_prompt}\n\n[USER]\n{local_user_prompt}"
    result.prompt_preview = (
        "Codegen prompt: healthcare denials analysis task + tool schema "
        "(Claude writes Python, host executes locally)"
    )
    result.integration_snippet = _local_integration_snippet(model)

    try:
        StateGraph, START, END = _langgraph_imports()
        ChatAnthropic = _langchain_anthropic_import()

        class LocalState(dict):
            pass

        def gen_code_node(state: LocalState) -> dict[str, Any]:
            next_state = dict(state)
            system_prompt, user_prompt = local_system_prompt, local_user_prompt
            logger.log("[LOCAL] LangGraph node: generate_code")
            llm = ChatAnthropic(model=model, max_tokens=1200, temperature=0)
            logger.log(f"[LOCAL] Calling Claude via LangChain model={model}")
            _append_transcript(result, "[LOCAL][TX][CODEGEN][SYSTEM]", system_prompt)
            _append_transcript(result, "[LOCAL][TX][CODEGEN][USER]", user_prompt)
            msg = llm.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            code = strip_code_fences(_extract_ai_text_from_langchain_message(msg))
            _append_transcript(result, "[LOCAL][RX][CODEGEN][ASSISTANT]", _extract_ai_text_from_langchain_message(msg))
            usage = _extract_usage_from_langchain_message(msg)
            _merge_usage(result.token_usage, usage, logger, "local_codegen")
            if "healthcare_claims_get_denials" not in code:
                raise RuntimeError(
                    "Generated code did not reference healthcare_claims_get_denials."
                )
            logger.log("[LOCAL] Claude generated Python code")
            for line in code.splitlines():
                logger.log(f"[LOCAL] [GENERATED CODE] {line}")
            next_state["generated_code"] = code
            return next_state

        def exec_node(state: LocalState) -> dict[str, Any]:
            next_state = dict(state)
            code = state.get("generated_code")
            if not isinstance(code, str) or not code.strip():
                raise RuntimeError("Local graph state missing generated_code before execution.")
            logger.log("[LOCAL] LangGraph node: execute_code")
            logger.log("[LOCAL] Executing Claude-generated Python locally via exec()")
            os.environ["DEMO_MODE"] = "LOCAL"

            # Capture generated code prints + tool prints into the structured log.
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec_globals = {
                    "__builtins__": __builtins__,
                    "healthcare_claims_get_denials": healthcare_claims_get_denials,
                }
                exec(code, exec_globals, {})
            out_text = stdout.getvalue()
            for line in out_text.splitlines():
                logger.log(line)
            next_state["execution_stdout"] = out_text
            return next_state

        def summarize_node(state: LocalState) -> dict[str, Any]:
            next_state = dict(state)
            logger.log("[LOCAL] LangGraph node: summarize_output")
            llm = ChatAnthropic(model=model, max_tokens=500, temperature=0)
            execution_stdout = state.get("execution_stdout", "")
            summary_prompt = _local_summary_prompt(str(execution_stdout))
            _append_transcript(result, "[LOCAL][TX][SUMMARY][USER]", summary_prompt)
            msg = llm.invoke([("human", summary_prompt)])
            final_text = _extract_ai_text_from_langchain_message(msg).strip()
            _append_transcript(result, "[LOCAL][RX][SUMMARY][ASSISTANT]", _extract_ai_text_from_langchain_message(msg))
            usage = _extract_usage_from_langchain_message(msg)
            _merge_usage(result.token_usage, usage, logger, "local_summary")
            logger.log("[LOCAL] Final assistant summary generated")
            next_state["final_text"] = final_text
            return next_state

        graph = StateGraph(dict)
        graph.add_node("generate_code", gen_code_node)
        graph.add_node("execute_code", exec_node)
        graph.add_node("summarize_output", summarize_node)
        graph.add_edge(START, "generate_code")
        graph.add_edge("generate_code", "execute_code")
        graph.add_edge("execute_code", "summarize_output")
        graph.add_edge("summarize_output", END)
        app = graph.compile()

        state = app.invoke({})
        result.generated_code = state.get("generated_code")
        result.execution_stdout = state.get("execution_stdout")
        result.final_text = state.get("final_text")
        result.metadata["dates"] = DATES
        result.metadata["model"] = model
        result.ok = True
    except Exception as exc:  # pragma: no cover - runtime/dependency/network
        result.error = f"{type(exc).__name__}: {exc}"
        logger.log(f"[LOCAL] ERROR: {result.error}", level="error")
        tb = traceback.format_exc(limit=8)
        for line in tb.splitlines():
            logger.log(f"[LOCAL] {line}", level="debug")
    finally:
        result.duration_ms = int(time.time() * 1000) - started
    return result


def _sandbox_tools(code_exec_tool_type: str) -> list[dict[str, Any]]:
    return [
        {"type": code_exec_tool_type, "name": "code_execution"},
        {
            "name": "healthcare_claims_get_denials",
            "description": (
                "Returns denied healthcare claims for a service_date in the demo dataset. "
                "Each row has keys: claim_id, patient_id, payer, denial_reason, denied_amount_NEW."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"service_date": {"type": "string"}},
                "required": ["service_date"],
                "additionalProperties": False,
            },
            "allowed_callers": [code_exec_tool_type],
        },
    ]


def _sandbox_prompt() -> str:
    return (
        "You are a healthcare revenue-cycle analyst assistant.\n\n"
        "Use the code execution tool to analyze denied claims for service dates "
        "2026-02-18, 2026-02-19, and 2026-02-20.\n"
        "Inside the executed Python code, use healthcare_claims_get_denials(service_date).\n"
        "Rows returned by the tool include the amount field named denied_amount_NEW "
        "(use that exact key for calculations).\n"
        "Make the run observable with [SANDBOX] print logs.\n"
        "Do not print every row; print counts and summaries only.\n"
        "Compute and print:\n"
        "- total denied amount across all dates\n"
        "- denied amount by denial_reason\n"
        "- denied amount by payer\n"
        "- top denial reason by denied amount\n\n"
        "First, show the full Python code in a plain text response (or fenced code block). "
        "Then execute it with the code execution tool. "
        "Choose the Python code structure yourself."
    )


def _extract_code_block_from_text(text: str) -> str | None:
    stripped = text.strip()
    if "```" not in stripped:
        return None
    parts = stripped.split("```")
    if len(parts) < 3:
        return None
    candidate = parts[1]
    lines = candidate.splitlines()
    if lines and lines[0].strip().lower().startswith("python"):
        lines = lines[1:]
    code = "\n".join(lines).strip()
    return code or None


def _get_container_id(response: dict[str, Any]) -> str | None:
    container = response.get("container")
    if isinstance(container, dict):
        return container.get("id")
    if isinstance(container, str):
        return container
    return None


def _log_sandbox_blocks(response: dict[str, Any], result: DemoResult, logger: Logger) -> dict[str, Any]:
    info: dict[str, Any] = {"pending_tool_calls": [], "remote_code": None}
    logger.log(f"[SANDBOX] API stop_reason={response.get('stop_reason')!r}")
    usage = response.get("usage")
    if isinstance(usage, dict):
        _merge_usage(result.token_usage, {"raw_usage": usage}, logger, "sandbox_turn")
    block_types = [b.get("type") for b in response.get("content", [])]
    logger.log(f"[SANDBOX] API content block types={block_types}")
    for idx, block in enumerate(response.get("content", []), start=1):
        btype = block.get("type")
        if btype == "server_tool_use":
            logger.log(
                f"[SANDBOX] Block {idx}: server_tool_use name={block.get('name')!r} id={block.get('id')!r}"
            )
            binput = block.get("input", {})
            if isinstance(binput, dict) and "code" in binput:
                code = str(binput["code"])
                info["remote_code"] = code
                logger.log(f"[SANDBOX] Block {idx}: generated remote Python code follows")
                for line in code.splitlines():
                    logger.log(f"[SANDBOX] [REMOTE CODE] {line}")
        elif btype == "code_execution_tool_result":
            stdout = block.get("stdout")
            stderr = block.get("stderr")
            if stdout:
                logger.log(f"[SANDBOX] Block {idx}: code_execution stdout follows")
                for line in str(stdout).splitlines():
                    logger.log(f"[SANDBOX] [REMOTE STDOUT] {line}")
            if stderr:
                logger.log(f"[SANDBOX] Block {idx}: code_execution stderr follows")
                for line in str(stderr).splitlines():
                    logger.log(f"[SANDBOX] [REMOTE STDERR] {line}")
        elif btype == "tool_use":
            logger.log(
                f"[SANDBOX] Block {idx}: tool_use name={block.get('name')!r} caller={block.get('caller')!r} id={block.get('id')!r}"
            )
            info["pending_tool_calls"].append(block)
        elif btype == "text":
            text = str(block.get("text", "")).strip()
            if text:
                logger.log(f"[SANDBOX] Block {idx}: text={text[:300]!r}")
                if info["remote_code"] is None:
                    maybe_code = _extract_code_block_from_text(text)
                    if maybe_code:
                        info["remote_code"] = maybe_code
                        logger.log(
                            f"[SANDBOX] Block {idx}: extracted remote Python code from text block"
                        )
    return info


def run_sandbox_langgraph(model: str | None = None) -> DemoResult:
    load_dotenv_if_present()
    model = model or DEFAULT_MODEL
    code_exec_tool_type = DEFAULT_CODE_EXEC_TOOL_TYPE
    started = int(time.time() * 1000)
    result = DemoResult(
        mode="sandbox_langgraph", ok=False, started_at_ms=started, duration_ms=0
    )
    logger = Logger(result, prefix="[SANDBOX]")
    logger.log("[SANDBOX] Mode A starting (LangGraph + Anthropic managed sandbox)")
    sandbox_prompt = _sandbox_prompt()
    result.prompt_text = sandbox_prompt
    result.prompt_preview = (
        "Sandbox prompt: ask Claude to show code + execute via code_execution tool "
        "with healthcare_claims_get_denials(...)"
    )
    result.integration_snippet = _sandbox_integration_snippet(model, code_exec_tool_type)

    try:
        StateGraph, START, END = _langgraph_imports()

        def setup_node(state: dict[str, Any]) -> dict[str, Any]:
            logger.log("[SANDBOX] LangGraph node: setup")
            next_state = dict(state)
            next_state.update({
                "conversation": [{"role": "user", "content": sandbox_prompt}],
                "tools": _sandbox_tools(code_exec_tool_type),
                "container_id": None,
                "turn": 0,
                "done": False,
                "assistant_texts": [],
                "pending_tool_calls": [],
                "remote_code": None,
                "final_text": None,
                "last_sent_count": 0,
            })
            return next_state

        def api_turn_node(state: dict[str, Any]) -> dict[str, Any]:
            turn = int(state.get("turn", 0)) + 1
            conversation_in = state.get("conversation")
            if not isinstance(conversation_in, list):
                conversation_in = [{"role": "user", "content": sandbox_prompt}]
            tools_in = state.get("tools")
            if not isinstance(tools_in, list):
                tools_in = _sandbox_tools(code_exec_tool_type)
                logger.log(
                    "[SANDBOX] tools missing from graph state; rebuilding tool definitions",
                    level="warning",
                )
            payload: dict[str, Any] = {
                "model": model,
                "max_tokens": 2200,
                "messages": conversation_in,
                "tools": tools_in,
            }
            last_sent_count = int(state.get("last_sent_count", 0))
            outbound_delta = conversation_in[last_sent_count:]
            if outbound_delta:
                _append_transcript(
                    result,
                    f"[SANDBOX][TX][TURN {turn}][MESSAGES_DELTA]",
                    outbound_delta,
                )
            if turn == 1:
                _append_transcript(result, "[SANDBOX][TX][TOOLS]", tools_in)
            if state.get("container_id"):
                payload["container"] = state["container_id"]
                logger.log(f"[SANDBOX] Reusing container={state['container_id']}")
            logger.log(f"[SANDBOX] LangGraph node: api_turn (turn={turn})")
            response = logger.capture_stdout_call(post_messages, payload, "[SANDBOX]")
            container_id = _get_container_id(response) or state.get("container_id")
            if container_id and not state.get("container_id"):
                logger.log("[SANDBOX] Container created")
                logger.log(f"[SANDBOX] Container id={container_id}")
            info = _log_sandbox_blocks(response, result, logger)
            _append_transcript(
                result,
                f"[SANDBOX][RX][TURN {turn}][CONTENT_BLOCKS]",
                _content_blocks_as_jsonable(response),
            )
            conversation = list(conversation_in)
            conversation.append({"role": "assistant", "content": response.get("content", [])})
            assistant_texts = list(state.get("assistant_texts", []))
            turn_texts = extract_text_blocks(response)
            if turn_texts:
                assistant_texts.append("\n".join(turn_texts).strip())
            next_state: dict[str, Any] = dict(state)
            next_state.update({
                "turn": turn,
                "response": response,
                "conversation": conversation,
                "container_id": container_id,
                "pending_tool_calls": info["pending_tool_calls"],
                "assistant_texts": assistant_texts,
                "tools": tools_in,
                "last_sent_count": len(conversation_in),
            })
            if info.get("remote_code"):
                next_state["remote_code"] = info["remote_code"]
            if not info["pending_tool_calls"]:
                next_state["done"] = True
                final_texts = extract_text_blocks(response)
                final_text = "\n".join(final_texts).strip()
                if not final_text:
                    final_text = "\n\n".join(t for t in assistant_texts if t).strip()
                next_state["final_text"] = final_text
            return next_state

        def handle_tools_node(state: dict[str, Any]) -> dict[str, Any]:
            logger.log("[SANDBOX] LangGraph node: handle_tools")
            tool_results: list[dict[str, Any]] = []
            for block in state.get("pending_tool_calls", []):
                if block.get("name") != "healthcare_claims_get_denials":
                    logger.log(
                        f"[SANDBOX] Skipping unexpected tool request {block.get('name')!r}",
                        level="warning",
                    )
                    continue
                tool_input = block.get("input", {}) or {}
                service_date = str(tool_input.get("service_date", ""))
                logger.log("[SANDBOX] Execution paused for tool result")
                logger.log(
                    "[SANDBOX] Intercepted tool call: "
                    f"healthcare_claims_get_denials(service_date={service_date!r})"
                )
                os.environ["DEMO_MODE"] = "SANDBOX"
                rows = logger.capture_stdout_call(
                    healthcare_claims_get_denials, service_date
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": json.dumps(rows),
                    }
                )
                result.metadata.setdefault("tool_calls", 0)
                result.metadata["tool_calls"] += 1
            conversation = list(state.get("conversation", []))
            conversation.append({"role": "user", "content": tool_results})
            logger.log("[SANDBOX] Execution resumed")
            next_state = dict(state)
            next_state.update({"conversation": conversation, "pending_tool_calls": []})
            return next_state

        def route_after_api(state: dict[str, Any]) -> str:
            return "handle_tools" if state.get("pending_tool_calls") else "done"

        graph = StateGraph(dict)
        graph.add_node("setup", setup_node)
        graph.add_node("api_turn", api_turn_node)
        graph.add_node("handle_tools", handle_tools_node)
        graph.add_edge(START, "setup")
        graph.add_edge("setup", "api_turn")
        graph.add_conditional_edges(
            "api_turn",
            route_after_api,
            {"handle_tools": "handle_tools", "done": END},
        )
        graph.add_edge("handle_tools", "api_turn")
        app = graph.compile()

        state = app.invoke({})
        result.generated_code = state.get("remote_code")
        result.final_text = state.get("final_text")
        if not result.final_text:
            # Fallback so the UI is never empty if the run otherwise succeeded.
            result.final_text = "No final assistant text returned. Check remote stdout in logs."
        result.metadata["model"] = model
        result.metadata["container_id"] = state.get("container_id")
        result.metadata["dates"] = DATES
        result.ok = True
    except Exception as exc:  # pragma: no cover
        result.error = f"{type(exc).__name__}: {exc}"
        logger.log(f"[SANDBOX] ERROR: {result.error}", level="error")
        tb = traceback.format_exc(limit=12)
        for line in tb.splitlines():
            logger.log(f"[SANDBOX] {line}", level="debug")
    finally:
        result.duration_ms = int(time.time() * 1000) - started
    return result


def run_traditional(model: str | None = None) -> DemoResult:
    """Traditional approach: classic tool calling (no code execution/programmatic tool calling)."""
    load_dotenv_if_present()
    model = model or DEFAULT_MODEL
    started = int(time.time() * 1000)
    result = DemoResult(mode="traditional", ok=False, started_at_ms=started, duration_ms=0)
    logger = Logger(result, prefix="[TRADITIONAL]")
    logger.log("[TRADITIONAL] Starting (classic tool calling; no programmatic tool calling)")
    logger.log(
        "[TRADITIONAL] Claude will call the business tool directly; no code_execution tool is used"
    )
    result.prompt_preview = (
        "Traditional prompt: classic tool calling only (Claude calls healthcare_claims_get_denials directly)"
    )
    traditional_prompt = _traditional_user_prompt()
    result.prompt_text = traditional_prompt
    result.generated_code = "N/A - Traditional mode uses direct tool calls (no generated Python code)."
    result.integration_snippet = _traditional_integration_snippet(model)
    result.llm_transcript = None
    try:
        tools = [
            {
                "name": "healthcare_claims_get_denials",
                "description": (
                    "Returns denied healthcare claims for a service_date in the demo dataset. "
                    "Each row has keys: claim_id, patient_id, payer, denial_reason, denied_amount_NEW."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"service_date": {"type": "string"}},
                    "required": ["service_date"],
                    "additionalProperties": False,
                },
            }
        ]
        conversation: list[dict[str, Any]] = [{"role": "user", "content": traditional_prompt}]
        assistant_texts: list[str] = []
        last_sent_count = 0

        for turn in range(1, 8):
            logger.log(f"[TRADITIONAL] API turn {turn} (classic tool loop)")
            outbound_delta = conversation[last_sent_count:]
            if outbound_delta:
                _append_transcript(
                    result,
                    f"[TRADITIONAL][TX][TURN {turn}][MESSAGES_DELTA]",
                    outbound_delta,
                )
            if turn == 1:
                _append_transcript(result, "[TRADITIONAL][TX][TOOLS]", tools)
            payload = {
                "model": model,
                "max_tokens": 1200,
                "messages": conversation,
                "tools": tools,
            }
            response = logger.capture_stdout_call(post_messages, payload, "[TRADITIONAL]")
            _append_transcript(
                result,
                f"[TRADITIONAL][RX][TURN {turn}][CONTENT_BLOCKS]",
                _content_blocks_as_jsonable(response),
            )
            usage = response.get("usage")
            if isinstance(usage, dict):
                _merge_usage(result.token_usage, {"raw_usage": usage}, logger, "traditional_turn")

            block_types = [b.get("type") for b in response.get("content", [])]
            logger.log(f"[TRADITIONAL] API content block types={block_types}")
            conversation.append({"role": "assistant", "content": response.get("content", [])})
            last_sent_count = len(conversation) - 1

            turn_texts = extract_text_blocks(response)
            if turn_texts:
                assistant_texts.append("\n".join(turn_texts).strip())

            tool_results: list[dict[str, Any]] = []
            for idx, block in enumerate(response.get("content", []), start=1):
                if block.get("type") == "text":
                    text = str(block.get("text", "")).strip()
                    if text:
                        logger.log(f"[TRADITIONAL] Block {idx}: text={text[:300]!r}")
                    continue
                if block.get("type") != "tool_use":
                    logger.log(
                        f"[TRADITIONAL] Block {idx}: {block.get('type')}",
                        level="debug",
                    )
                    continue
                if block.get("name") != "healthcare_claims_get_denials":
                    logger.log(
                        f"[TRADITIONAL] Unexpected tool request: {block.get('name')!r}",
                        level="warning",
                    )
                    continue
                tool_input = block.get("input", {}) or {}
                service_date = str(tool_input.get("service_date", ""))
                logger.log(
                    "[TRADITIONAL] Tool call: "
                    f"healthcare_claims_get_denials(service_date={service_date!r})"
                )
                os.environ["DEMO_MODE"] = "TRADITIONAL"
                rows = logger.capture_stdout_call(healthcare_claims_get_denials, service_date)
                result.metadata.setdefault("tool_calls", 0)
                result.metadata["tool_calls"] += 1
                result.metadata.setdefault("rows_returned_total", 0)
                result.metadata["rows_returned_total"] += len(rows)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": json.dumps(rows),
                    }
                )

            if tool_results:
                conversation.append({"role": "user", "content": tool_results})
                continue

            result.final_text = "\n\n".join(t for t in assistant_texts if t).strip()
            if not result.final_text:
                result.final_text = "No final assistant text returned."
            break
        else:
            raise RuntimeError("Traditional tool-calling loop exceeded max turns.")

        result.metadata["model"] = model
        result.metadata["dates"] = DATES
        result.ok = True
    except Exception as exc:  # pragma: no cover
        result.error = f"{type(exc).__name__}: {exc}"
        logger.log(f"[TRADITIONAL] ERROR: {result.error}", level="error")
        tb = traceback.format_exc(limit=8)
        for line in tb.splitlines():
            logger.log(f"[TRADITIONAL] {line}", level="debug")
    finally:
        result.duration_ms = int(time.time() * 1000) - started
    return result


def run_mode(mode: str, model: str | None = None) -> DemoResult:
    if mode == "local_langgraph":
        return run_local_langgraph(model=model)
    if mode == "sandbox_langgraph":
        return run_sandbox_langgraph(model=model)
    if mode == "traditional":
        return run_traditional(model=model)
    started = int(time.time() * 1000)
    result = DemoResult(mode=mode, ok=False, started_at_ms=started, duration_ms=0)
    result.error = f"Unknown mode: {mode}"
    result.duration_ms = int(time.time() * 1000) - started
    return result
