"""LangGraph pipeline implementing the Symbioza CoT agent."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict, Union

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .memory.memory_utils import fetch_relevant
from .tools import Toolset

ALLOWED_INTENTS = {"ask_coder", "notify_planner", "broadcast_status", "status_reply"}
DEFAULT_CODER_TARGET = "symbioza-aider"
TRACE_LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "agent_traces.jsonl"
TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


class SendMessageAction(BaseModel):
    type: Literal["SendMessage"] = "SendMessage"
    module: str
    payload: Dict[str, Any]


class BroadcastAction(BaseModel):
    type: Literal["Broadcast"] = "Broadcast"
    modules: List[str]
    payload: Dict[str, Any]


class ExecCommandAction(BaseModel):
    type: Literal["ExecCommand"] = "ExecCommand"
    instruction: str
    files: List[str]


Action = Union[SendMessageAction, BroadcastAction, ExecCommandAction]


class CoTState(TypedDict, total=False):
    sender: str
    payload: Dict[str, Any]
    history: List[str]
    memory_context: List[Dict[str, Any]]
    thoughts: List[str]
    intent: str
    targets: List[str]
    todo: str
    actions: List[Action]
    action_results: List[Dict[str, Any]]
    trace: Dict[str, Any]
    delivered_to: List[str]


def _summarise_payload(sender: str, payload: Dict[str, Any]) -> str:
    topic = payload.get("topic") or payload.get("summary")
    if topic:
        return f"{sender} reports: {topic}"
    if "instruction" in payload:
        return f"{sender} requests code changes."
    if payload:
        keys = ", ".join(sorted(payload.keys())[:4])
        return f"{sender} shared payload keys: {keys}"
    return f"{sender} pinged the coordinator."


def _think_node(state: CoTState) -> Dict[str, Any]:
    sender = state.get("sender", "unknown")
    payload = state.get("payload", {})
    history = state.get("history") or []

    query_source = payload if payload else {"sender": sender}
    try:
        query_text = json.dumps(query_source)
    except TypeError:
        query_text = str(query_source)

    try:
        memory_context = fetch_relevant(query_text, k=3)
        print(f"[MEMORY] Context loaded ({len(memory_context)} items)")
    except Exception as exc:  # noqa: BLE001
        print(f"[MEMORY] Context load failed: {exc}")
        memory_context = []

    requested_intent = payload.get("intent")
    intent = requested_intent if requested_intent in ALLOWED_INTENTS else None

    if intent is None:
        if payload.get("instruction") or payload.get("needs_code"):
            intent = "ask_coder"
        elif payload.get("broadcast"):
            intent = "broadcast_status"
        elif payload.get("notify"):
            intent = "notify_planner"
        else:
            intent = "status_reply"

    targets: List[str] = []
    payload_targets = payload.get("targets")
    if isinstance(payload_targets, list) and payload_targets:
        targets = [str(t) for t in payload_targets]
    elif payload.get("target"):
        targets = [str(payload["target"])]
    elif intent == "ask_coder":
        targets = [DEFAULT_CODER_TARGET]
    elif intent == "status_reply":
        targets = [sender]

    todo = payload.get("todo") or payload.get("instruction") or "Acknowledge the request."
    thoughts = [
        _summarise_payload(sender, payload),
        f"Intent: {intent}",
    ]
    if memory_context:
        memory_hint = memory_context[0].get("summary") if isinstance(memory_context[0], dict) else None
        hint_text = memory_hint or f"{len(memory_context)} memory entries available."
        thoughts.insert(0, f"Memory recall: {hint_text}")
    if targets:
        thoughts.append(f"Targets: {', '.join(targets)}")
    if len(thoughts) > 3:
        thoughts = thoughts[:3]

    if history:
        thoughts[-1] = f"History noted ({len(history)} items)."

    return {"thoughts": thoughts, "intent": intent, "targets": targets, "todo": todo, "memory_context": memory_context}


def _route_node(state: CoTState) -> Dict[str, Any]:
    intent = state.get("intent", "status_reply")
    payload = state.get("payload", {})
    sender = state.get("sender", "")
    targets = state.get("targets", []) or []

    actions: List[Action] = []

    if intent == "ask_coder":
        instruction = str(payload.get("instruction") or state.get("todo") or "Assist with request.")
        files_payload = payload.get("files") or []
        files = [str(f) for f in files_payload if isinstance(f, str)]
        actions.append(ExecCommandAction(instruction=instruction, files=files))
        ack_payload = {"status": "delegated", "executor": DEFAULT_CODER_TARGET, "todo": instruction}
        actions.append(SendMessageAction(module=sender, payload=ack_payload))
    elif intent == "notify_planner":
        target = targets[0] if targets else sender
        message_payload = payload.get("message") or {"status": "update", "todo": state.get("todo", "")}
        actions.append(SendMessageAction(module=target, payload=message_payload))
    elif intent == "broadcast_status":
        modules = targets or [sender]
        broadcast_payload = payload.get("message") or {"status": "broadcast", "from": sender}
        actions.append(BroadcastAction(modules=modules, payload=broadcast_payload))
    else:
        reply_payload = payload.get("reply") or {"status": "ok", "detail": state.get("todo", "")}
        actions.append(SendMessageAction(module=sender, payload=reply_payload))

    return {"actions": actions}


def _jsonify_node(state: CoTState) -> Dict[str, Any]:
    actions = state.get("actions") or []
    trace = {
        "from": state.get("sender"),
        "thoughts": state.get("thoughts", []),
        "intent": state.get("intent"),
        "targets": state.get("targets", []),
        "actions": [action.model_dump() for action in actions],
    }
    return {"trace": trace}


async def _write_trace(trace: Dict[str, Any]) -> None:
    line = json.dumps(trace, ensure_ascii=False)
    await asyncio.to_thread(_append_trace_line, line)


def _append_trace_line(line: str) -> None:
    with TRACE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _trace_event(event: str, payload: Dict[str, Any]) -> None:
    record = {"event": event, "timestamp": _timestamp(), **payload}
    await _write_trace(record)


async def _emit_node(state: CoTState, tools: Toolset) -> Dict[str, Any]:
    sender = state.get("sender")
    delivered = set(state.get("delivered_to") or [])
    actions = state.get("actions") or []

    sender_in_actions = any(
        (
            isinstance(action, SendMessageAction) and action.module == sender
        )
        or (
            isinstance(action, BroadcastAction) and sender in action.modules
        )
        for action in actions
    )

    if sender and sender_in_actions and sender not in delivered:
        ack_payload = {"status": "sent", "detail": state.get("todo", "")}
        try:
            await tools.messaging.send(sender, ack_payload)
            delivered.add(sender)
        except Exception:  # pragma: no cover - logging handled by caller
            pass

    trace = state.get("trace")
    if trace:
        trace_with_delivery = dict(trace)
        trace_with_delivery["delivered_to"] = sorted(delivered)
        await _write_trace(trace_with_delivery)

    return {"delivered_to": sorted(delivered)}


def build_cot_graph(tools: Toolset):
    """Compile the LangGraph DAG for the CoT agent."""

    graph = StateGraph(CoTState)
    graph.add_node("think", _think_node)
    graph.add_node("route", _route_node)

    async def act(state: CoTState) -> Dict[str, Any]:
        return await _act_node(state, tools)

    graph.add_node("act", act)
    graph.add_node("jsonify", _jsonify_node)

    async def emit(state: CoTState) -> Dict[str, Any]:
        return await _emit_node(state, tools)

    graph.add_node("emit", emit)

    graph.add_edge(START, "think")
    graph.add_edge("think", "route")
    graph.add_edge("route", "act")
    graph.add_edge("act", "jsonify")
    graph.add_edge("jsonify", "emit")
    graph.add_edge("emit", END)

    return graph.compile()


async def _handle_exec_action(
    action: ExecCommandAction,
    tools: Toolset,
) -> Dict[str, Any]:
    redis = getattr(tools, "redis", None)

    print(f"[COT] ExecCommand started: {action.instruction}")
    await _trace_event(
        "exec_start",
        {
            "instruction": action.instruction,
            "files": action.files,
        },
    )

    try:
        result = await tools.executor.aider(action.instruction, action.files)
    except Exception as exc:
        print(f"[COT] Executor offline: {exc}")
        await _trace_event(
            "executor_offline",
            {
                "instruction": action.instruction,
                "files": action.files,
                "error": str(exc),
            },
        )
        warning_payload = {
            "from": "symbioza-cot",
            "status": "warning",
            "commit": "none",
            "detail": "Executor offline, queued for later",
            "timestamp": _timestamp(),
        }
        try:
            await tools.messaging.send("owner", warning_payload)
            await _trace_event(
                "executor_offline_notify",
                {"module": "owner", "payload": warning_payload},
            )
        except Exception as notify_exc:  # noqa: BLE001
            print(f"[COT] Failed to notify owner about offline executor: {notify_exc}")
            await _trace_event(
                "executor_offline_notify_error",
                {"module": "owner", "payload": warning_payload, "error": str(notify_exc)},
            )
        if redis is not None:
            try:
                await redis.xadd("symbioza:msg:owner", {k: str(v) for k, v in warning_payload.items()})
            except Exception as redis_exc:  # noqa: BLE001
                print(f"[COT] Failed to enqueue offline warning: {redis_exc}")
                await _trace_event(
                    "executor_offline_enqueue_error",
                    {"stream": "symbioza:msg:owner", "payload": warning_payload, "error": str(redis_exc)},
                )
        return {"status": "warning", "detail": "Executor offline", "error": str(exc)}

    commit = result.get("commit", "")
    print(f"[COT] ExecCommand finished: commit={commit}")
    await _trace_event(
        "exec_success",
        {
            "instruction": action.instruction,
            "files": action.files,
            "result": result,
        },
    )
    if redis is not None:
        payload = {
            "from": "symbioza-cot",
            "status": "done",
            "commit": commit or "unknown",
            "detail": "Aider exec complete",
            "timestamp": _timestamp(),
        }
        try:
            await redis.xadd("symbioza:msg:owner", {k: str(v) for k, v in payload.items()})
            await _trace_event(
                "exec_notify",
                {"stream": "symbioza:msg:owner", "payload": payload},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[COT] Failed to notify owner: {exc}")
            await _trace_event(
                "exec_notify_error",
                {"stream": "symbioza:msg:owner", "payload": payload, "error": str(exc)},
            )
    return result


async def _act_node(state: CoTState, tools: Toolset) -> Dict[str, Any]:
    actions = state.get("actions") or []
    results: List[Dict[str, Any]] = []
    delivered: List[str] = state.get("delivered_to", [])
    delivered_set = set(delivered)

    for action in actions:
        if isinstance(action, SendMessageAction):
            outcome = await tools.messaging.send(action.module, action.payload)
            delivered_set.add(action.module)
        elif isinstance(action, BroadcastAction):
            outcome = await tools.messaging.broadcast(action.modules, action.payload)
            for module in action.modules:
                delivered_set.add(module)
        elif isinstance(action, ExecCommandAction):
            try:
                outcome = await _handle_exec_action(action, tools)
            except Exception as exc:  # noqa: BLE001
                outcome = {"status": "error", "detail": str(exc)}
        else:  # pragma: no cover - safeguarding unexpected action
            continue
        results.append({"action": action.model_dump(), "result": outcome})

    return {"action_results": results, "delivered_to": sorted(delivered_set)}
