from typing import List, Dict
import json
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
STM_FILE = MEMORY_DIR / "short_term.jsonl"
LTM_FILE = MEMORY_DIR / "long_term.jsonl"


def append_event(event: Dict, file: Path = STM_FILE):
    """Append a new memory event to short_term.jsonl"""
    event["timestamp"] = datetime.utcnow().isoformat()
    with open(file, "a") as f:
        f.write(json.dumps(event) + "\n")


def fetch_recent(limit: int = 20) -> List[Dict]:
    """Return the N most recent memory events"""
    if not STM_FILE.exists():
        return []
    with open(STM_FILE) as f:
        lines = f.readlines()[-limit:]
    return [json.loads(l) for l in lines]


def summarize() -> str:
    """Summarize short-term memory into one long-term note"""
    stm = fetch_recent(50)
    if not stm:
        return "No recent events."
    summary = "Summary of recent activity:\n" + "\n".join(
        [f"- {e.get('from','unknown')}: {e.get('payload',{})}" for e in stm]
    )
    append_event({"summary": summary}, file=LTM_FILE)
    return summary


def fetch_relevant(query: str, k: int = 3) -> List[Dict]:
    """Naive relevance search: return up to k entries containing the query."""
    if not LTM_FILE.exists():
        return []
    with open(LTM_FILE) as f:
        entries = [json.loads(l) for l in f.readlines()]
    return [e for e in entries if query.lower() in json.dumps(e).lower()][:k]


def load_recent_error_reports(limit: int = 10) -> List[Dict]:
    """Return the most recent ErrorReport entries from long-term memory."""
    if not LTM_FILE.exists() or limit <= 0:
        return []

    entries: List[Dict] = []
    with open(LTM_FILE) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(record, dict):
                payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
                if isinstance(payload, dict) and payload.get("type") == "ErrorReport":
                    entries.append(record)

    return entries[-limit:]
