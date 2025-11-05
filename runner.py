"""Async runner that consumes Redis streams and executes the CoT graph."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

try:
    import uvloop
except ImportError:  # pragma: no cover - optional dependency
    uvloop = None
from redis.asyncio import Redis
from redis.exceptions import RedisError

from modules.symbioza_core.config.loader import get_effective_config

from .cot_graph import TRACE_LOG_PATH, build_cot_graph
from .memory.memory_utils import append_event, summarize
from .tools import ExecTool, MessagingTool, Toolset

INBOX_STREAM = "symbioza:msg:symbioza-cot"
IDLE_SLEEP_SECONDS = 3.0
_MEMORY_COUNT = 0


def _env_var(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


async def _record_memory(sender: str, decoded: Dict[str, Any]) -> None:
    global _MEMORY_COUNT
    _MEMORY_COUNT += 1

    event = {
        "from": sender,
        "payload": decoded,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        await asyncio.to_thread(append_event, event)
        print(f"[MEMORY] STM append success (count={_MEMORY_COUNT})")
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        print(f"[MEMORY] STM append failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[MEMORY] STM append failed: {exc}")

    if _MEMORY_COUNT % 10 == 0:
        try:
            summary = await asyncio.to_thread(summarize)
            print("[MEMORY] LTM summarize triggered")
            if summary:
                first_line = summary.splitlines()[0]
                print(f"[MEMORY] Summary note: {first_line}")
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            print(f"[MEMORY] LTM summarize failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[MEMORY] LTM summarize failed: {exc}")


async def _log_auto_review(stage: str, payload: Dict[str, Any]) -> None:
    record = {
        "event": "auto_review",
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    line = json.dumps(record, ensure_ascii=False)
    await asyncio.to_thread(_append_trace_line, line)


async def _process_event(graph, tools: Toolset, message_id: str, payload: Dict[str, str]) -> None:
    sender = payload.get("from", "unknown")
    raw_payload = payload.get("payload", "{}")
    try:
        decoded: Dict[str, Any] = json.loads(raw_payload)
    except json.JSONDecodeError:
        decoded = {"_raw": raw_payload}

    await _record_memory(sender, decoded)

    msg_type = decoded.get("type")
    if msg_type == "ExecResult":
        status_text = str(decoded.get("status", "")).lower()
        verification_status = "verified" if status_text in {"ok", "success"} else "failed"
        feedback_payload = {
            "type": "RepairFeedback",
            "status": verification_status,
            "detail": decoded.get("detail"),
            "commit": decoded.get("commit"),
            "origin": decoded.get("origin"),
        }
        try:
            await tools.messaging.send("symbioza-critic", feedback_payload)
            await _log_auto_review("cot_verify", {"status": verification_status, "feedback": feedback_payload})
        except Exception as exc:  # noqa: BLE001
            await _log_auto_review("cot_verify_error", {"error": str(exc)})

        if status_text in {"ok", "success"}:
            commit_id = decoded.get("commit") or decoded.get("commit_id", "")
            verification_payload = {
                "type": "VerificationResult",
                "status": "success",
                "commit": commit_id,
                "detail": "Auto-review passed",
            }
            try:
                redis_client = getattr(tools, "redis", None)
                if isinstance(redis_client, Redis):
                    body = {
                        "from": "symbioza-cot",
                        "payload": json.dumps(verification_payload),
                    }
                    await redis_client.xadd("symbioza:msg:symbioza-critic", body)
                    await _log_auto_review(
                        "cot_verification_sent",
                        {"commit": commit_id, "payload": verification_payload},
                    )
            except RedisError as exc:
                await _log_auto_review(
                    "cot_verification_error",
                    {"error": str(exc), "payload": verification_payload},
                )

    history = decoded.get("history")
    if not isinstance(history, list):
        history = []

    state = {
        "sender": sender,
        "payload": decoded,
        "history": history,
    }

    await graph.ainvoke(state)


def _append_trace_line(line: str) -> None:
    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


async def _log_message(message_id: str, payload: Dict[str, Any], stage: str) -> None:
    record = {
        "event": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id": message_id,
        "payload": payload,
    }
    line = json.dumps(record, ensure_ascii=False)
    await asyncio.to_thread(_append_trace_line, line)


async def _consume_loop(graph, redis: Redis, tools: Toolset, *, debug: bool) -> None:
    last_id = "0"
    catching_up = True
    if debug:
        print(f"[COT] Starting backlog replay from {last_id}")

    while True:
        try:
            if catching_up:
                response: List[Tuple[str, List[Tuple[str, Dict[str, str]]]]] = await redis.xread(
                    {INBOX_STREAM: last_id}, count=50
                )
            else:
                if debug:
                    print("[COT] Waiting for new messages...")
                response = await redis.xread({INBOX_STREAM: "$"}, block=5000, count=10)
        except RedisError:
            await asyncio.sleep(1.0)
            continue

        if not response:
            if catching_up:
                catching_up = False
                if debug:
                    print("[COT] Backlog drained; switching to live stream.")
            else:
                if debug:
                    print(f"[COT] Idle, sleeping for {IDLE_SLEEP_SECONDS:.0f}s.")
                await asyncio.sleep(IDLE_SLEEP_SECONDS)
            continue

        for _, messages in response:
            for message_id, payload in messages:
                if catching_up:
                    last_id = message_id
                if debug:
                    print(f"[COT] Received {message_id}: {payload}")
                await _log_message(message_id, payload, "incoming")
                try:
                    await _process_event(graph, tools, message_id, payload)
                except Exception:
                    continue
        if catching_up and response:
            continue
        catching_up = False


async def main(debug: bool = False) -> None:
    redis_host = _env_var("REDIS_HOST", "localhost")
    redis_port = int(_env_var("REDIS_PORT", "6379"))
    redis_db = int(_env_var("REDIS_DB", "0"))
    core_api_url = _env_var("CORE_API_URL", "http://127.0.0.1:8000")
    api_key = _env_var("SYM_API_KEY", "")

    cfg = get_effective_config("symbioza-cot")
    print(f"[CFG] Agent symbioza-cot using {cfg.get('model')} via {cfg.get('provider')}")

    if debug:
        print(f"[COT] Connecting to Redis at {redis_host}:{redis_port}/{redis_db}")
        print(f"[COT] Core API endpoint: {core_api_url}")

    redis = Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True, encoding="utf-8")
    tools = Toolset(
        messaging=MessagingTool(core_api_url, api_key),
        executor=ExecTool(core_api_url, api_key),
    )
    setattr(tools, "redis", redis)
    graph = build_cot_graph(tools)

    stop_event = asyncio.Event()

    def _signal_handler(*_: Any) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    if debug:
        print("[COT] Runner initialised; entering consume loop.")

    consumer = asyncio.create_task(_consume_loop(graph, redis, tools, debug=debug))

    await stop_event.wait()
    consumer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer

    await tools.messaging.aclose()
    await tools.executor.aclose()
    await redis.close()
    await redis.connection_pool.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symbioza CoT runner")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if uvloop is not None:
        uvloop.install()
    asyncio.run(main(debug=args.debug))
