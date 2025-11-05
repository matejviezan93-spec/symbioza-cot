# Symbioza CoT Module

LangGraph-based chain-of-thought (CoT) agent that reads module messages from Redis
streams, plans responses, executes tools, and emits structured traces.

## Requirements

Install dependencies inside your virtualenv:

```bash
pip install -r modules/symbioza-cot/requirements.txt
```

Environment variables:

| Name | Description | Default |
| --- | --- | --- |
| `REDIS_HOST` | Redis hostname | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_DB` | Redis database index | `0` |
| `CORE_API_URL` | Base URL for the Symbioza Core API | `http://127.0.0.1:8000` |
| `SYM_API_KEY` | API key for authenticated requests | _required_ |
| `OPENAI_API_KEY` | Optional, forwarded to downstream LLM tooling | _unset_ |

## Running the agent

```bash
python3 modules/symbioza-cot/runner.py
```

The runner listens on the Redis stream `symbioza:msg:symbioza-cot`, executes the CoT
DAG, and logs traces to `logs/agent_traces.jsonl`.

## Messaging walkthrough

Send a message _to_ the CoT agent:

```bash
curl -X POST "$CORE_API_URL/v1/message/send" \
  -H "x-api-key: $SYM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "from": "planner",
        "to": "symbioza-cot",
        "payload": {
          "intent": "notify_planner",
          "targets": ["symbioza-planner"],
          "message": {"status": "sync", "detail": "Need status update"}
        }
      }'
```

Fetch the response intended for `symbioza-planner`:

```bash
curl -X GET "$CORE_API_URL/v1/message/receive/symbioza-planner" \
  -H "x-api-key: $SYM_API_KEY"
```

## Testing

```bash
pytest -q tests/test_cot_graph.py
```
