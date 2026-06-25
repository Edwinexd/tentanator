"""Async HTTP client for the Tentanator backend API.

The TUI holds no grading logic of its own; every operation is a call into the
Rust Axum backend defined in ../ARCHITECTURE.md.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_BASE_URL = os.getenv("TENTANATOR_API", "http://127.0.0.1:8787")


class APIError(Exception):
    """Raised when the backend returns a non-2xx response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class TentanatorAPI:
    """Thin async wrapper over the backend's /api endpoints."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = await self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise APIError(0, f"cannot reach backend at {self.base_url} ({exc})") from exc
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("error", detail)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            raise APIError(resp.status_code, detail)
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # -- health & exams ----------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/health")

    async def list_exams(self) -> List[str]:
        return await self._request("GET", "/api/exams")

    async def exam_columns(self, file: str) -> List[str]:
        return await self._request("GET", f"/api/exams/{file}/columns")

    async def exam_rows(self, file: str) -> List[Dict[str, str]]:
        data = await self._request("GET", f"/api/exams/{file}/rows")
        return data.get("rows", [])

    # -- sessions ----------------------------------------------------------
    async def list_sessions(self, archived: bool = False) -> List[Dict[str, Any]]:
        return await self._request("GET", "/api/sessions", params={"archived": str(archived).lower()})

    async def create_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/api/sessions", json=payload)

    async def get_session(self, name: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/sessions/{name}")

    async def delete_session(self, name: str) -> None:
        await self._request("DELETE", f"/api/sessions/{name}")

    async def archive_session(self, name: str) -> None:
        await self._request("POST", f"/api/sessions/{name}/archive")

    # -- questions & grading ----------------------------------------------
    async def put_question(self, name: str, col: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("PUT", f"/api/sessions/{name}/questions/{col}", json=meta)

    async def sampling(self, name: str, col: str, algorithm: str,
                       n_samples: Optional[int] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"algorithm": algorithm}
        if n_samples is not None:
            body["n_samples"] = n_samples
        return await self._request("POST", f"/api/sessions/{name}/questions/{col}/sampling", json=body)

    async def grade(self, name: str, col: str, row_id: str, grade: str) -> Dict[str, Any]:
        return await self._request(
            "POST", f"/api/sessions/{name}/questions/{col}/grade",
            json={"row_id": row_id, "grade": grade},
        )

    async def ungrade(self, name: str, col: str, row_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/api/sessions/{name}/questions/{col}/grade/{row_id}")

    async def suggest(self, name: str, col: str, row_id: str) -> Dict[str, Any]:
        return await self._request(
            "POST", f"/api/sessions/{name}/questions/{col}/suggest",
            json={"row_id": row_id},
        )

    async def question_status(self, name: str, col: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/sessions/{name}/questions/{col}/status")

    async def export(self, name: str) -> Dict[str, Any]:
        return await self._request("POST", f"/api/sessions/{name}/export")
