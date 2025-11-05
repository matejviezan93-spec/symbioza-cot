"""HTTP tools used by the Symbioza CoT agent."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List, Protocol, runtime_checkable

import httpx

DEFAULT_MODULE_ID = "symbioza-cot"


@runtime_checkable
class MessagingToolProtocol(Protocol):
    async def send(self, module: str, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    async def broadcast(self, modules: List[str], payload: dict[str, Any]) -> List[dict[str, Any]]:
        ...


@runtime_checkable
class ExecToolProtocol(Protocol):
    async def aider(self, instruction: str, files: List[str]) -> dict[str, Any]:
        ...


@dataclass
class Toolset:
    messaging: MessagingToolProtocol
    executor: ExecToolProtocol


class MessagingTool(MessagingToolProtocol):
    """Wrapper around the messaging API with simple retry semantics."""

    def __init__(
        self,
        core_api_url: str,
        api_key: str,
        *,
        module_id: str = DEFAULT_MODULE_ID,
        timeout: float = 5.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=core_api_url.rstrip("/"),
            timeout=timeout,
            headers={"x-api-key": api_key},
        )
        self._module_id = module_id

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = await self._client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.1)
        raise RuntimeError(f"Messaging request failed: {last_error}") from last_error

    async def send(self, module: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = {"from": self._module_id, "to": module, "payload": payload}
        return await self._post("/v1/message/send", body)

    async def broadcast(self, modules: List[str], payload: dict[str, Any]) -> List[dict[str, Any]]:
        results: List[dict[str, Any]] = []
        for module in modules:
            results.append(await self.send(module, payload))
        return results

    async def aclose(self) -> None:
        await self._client.aclose()


class ExecTool(ExecToolProtocol):
    """Wrapper around the aider executor endpoint."""

    def __init__(
        self,
        core_api_url: str,
        api_key: str,
        *,
        timeout: float = 5.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=core_api_url.rstrip("/"),
            timeout=timeout,
            headers={"x-api-key": api_key},
        )

    async def aider(self, instruction: str, files: List[str]) -> dict[str, Any]:
        body = {"instruction": instruction, "files": files}
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = await self._client.post("/v1/exec/aider", json=body)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.1)
        raise RuntimeError(f"Executor request failed: {last_error}") from last_error

    async def aclose(self) -> None:
        await self._client.aclose()

