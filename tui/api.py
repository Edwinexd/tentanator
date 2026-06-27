"""Async HTTP client for the Tentanator backend API.

The TUI holds no grading logic of its own; every operation is a call into the
Rust Axum backend defined in ../ARCHITECTURE.md.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

DEFAULT_BASE_URL = os.getenv("TENTANATOR_API", "http://127.0.0.1:8787")


def _p(seg: Any) -> str:
    """Percent-encode a single path segment (mirrors the web client).

    Column names like ``"Response 1"`` and ids/names may contain spaces or
    ``/``; ``safe=""`` encodes every reserved character so the segment cannot
    leak into the URL structure.
    """
    return quote(str(seg), safe="")


class APIError(Exception):
    """Raised when the backend returns a non-2xx response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


# A 1:1 mirror of the backend's HTTP routes: every method is a trivial
# passthrough, so per-method docstrings add noise and the method count tracks
# the API surface by design (parity is enforced by scripts/check_api_parity.py).
# pylint: disable=missing-function-docstring,too-many-public-methods
class TentanatorAPI:
    """Thin async wrapper over the backend's /api endpoints."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
        # Export/download routes stream bytes the backend does not persist, so
        # the client saves them locally.
        self.graded_dir = os.getenv("TENTANATOR_GRADED_DIR", "graded_exams")

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _error_detail(resp: httpx.Response) -> str:
        detail = resp.text
        try:
            detail = resp.json().get("error", detail)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return detail

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            resp = await self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise APIError(0, f"cannot reach backend at {self.base_url} ({exc})") from exc
        if resp.status_code >= 400:
            raise APIError(resp.status_code, self._error_detail(resp))
        return resp

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = await self._send(method, path, **kwargs)
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    async def _download(self, method: str, path: str, **kwargs: Any) -> Dict[str, str]:
        """Fetch a file attachment and save it under ``graded_dir``.

        Export/download routes stream bytes (not JSON); the backend does not
        persist them. Returns ``{"filename", "path"}``.
        """
        resp = await self._send(method, path, **kwargs)
        cd = resp.headers.get("content-disposition", "")
        match = re.search(r'filename="?([^"]+)"?', cd)
        filename = match.group(1) if match else (os.path.basename(path) or "download")
        os.makedirs(self.graded_dir, exist_ok=True)
        out_path = os.path.join(self.graded_dir, filename)
        with open(out_path, "wb") as fh:
            fh.write(resp.content)
        return {"filename": filename, "path": out_path}

    # -- health & exam files ----------------------------------------------
    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/health")

    async def list_exam_files(self) -> List[str]:
        return await self._request("GET", "/api/exam-files")

    async def exam_columns(self, file: str) -> List[str]:
        return await self._request("GET", f"/api/exam-files/{_p(file)}/columns")

    async def exam_rows(self, file: str) -> List[Dict[str, str]]:
        data = await self._request("GET", f"/api/exam-files/{_p(file)}/rows")
        return data.get("rows", [])

    async def detect_columns(self, file: str) -> Dict[str, List[str]]:
        return await self._request("GET", f"/api/exam-files/{_p(file)}/detect")

    async def list_scans(self) -> List[str]:
        return await self._request("GET", "/api/scans")

    async def combine_moodle(self, grades_file: str, responses_file: str,
                             output_name: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/api/exam-files/combine-moodle", json={
            "grades_file": grades_file,
            "responses_file": responses_file,
            "output_name": output_name,
        })

    # -- exams (the central entity) ---------------------------------------
    async def list_exams(self, archived: bool = False) -> List[Dict[str, Any]]:
        return await self._request("GET", "/api/exams", params={"archived": str(archived).lower()})

    async def create_exam(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/api/exams", json=payload)

    async def get_exam(self, name: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/exams/{_p(name)}")

    async def update_exam(self, name: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("PUT", f"/api/exams/{_p(name)}", json=meta)

    async def delete_exam(self, name: str) -> None:
        await self._request("DELETE", f"/api/exams/{_p(name)}")

    async def archive_exam(self, name: str) -> None:
        await self._request("POST", f"/api/exams/{_p(name)}/archive")

    async def unarchive_exam(self, name: str) -> None:
        await self._request("POST", f"/api/exams/{_p(name)}/unarchive")

    async def update_exam_columns(self, name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("PUT", f"/api/exams/{_p(name)}/columns", json=body)

    # -- legacy import (old Python-app data -> new format, on demand) -------
    async def legacy_sessions_info(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/legacy-sessions")

    async def import_legacy_sessions(self) -> Dict[str, Any]:
        return await self._request("POST", "/api/legacy-sessions/import")

    async def list_legacy_workspaces(self) -> List[Dict[str, Any]]:
        return await self._request("GET", "/api/legacy-workspaces")

    async def import_workspace(self, name: str) -> Dict[str, Any]:
        return await self._request("POST", f"/api/legacy-workspaces/{_p(name)}/import")

    # -- sessions (grading passes under an exam) --------------------------
    async def list_sessions(self, exam: str) -> List[Dict[str, Any]]:
        return await self._request("GET", f"/api/exams/{_p(exam)}/sessions")

    async def create_session(self, exam: str, name: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", f"/api/exams/{_p(exam)}/sessions", json={"name": name})

    async def delete_session(self, exam: str, session: str) -> None:
        await self._request("DELETE", f"/api/exams/{_p(exam)}/sessions/{_p(session)}")

    # -- questions & grading ----------------------------------------------
    async def put_question(self, name: str, col: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request(
            "PUT", f"/api/exams/{_p(name)}/questions/{_p(col)}", json=meta)

    async def sampling(self, name: str, col: str, algorithm: str,
                       n_samples: Optional[int] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"algorithm": algorithm}
        if n_samples is not None:
            body["n_samples"] = n_samples
        return await self._request(
            "POST", f"/api/exams/{_p(name)}/questions/{_p(col)}/sampling", json=body)

    async def grade(self, name: str, col: str, row_id: str, grade: str,
                    session: Optional[str] = None) -> Dict[str, Any]:
        return await self._request(
            "POST", f"/api/exams/{_p(name)}/questions/{_p(col)}/grade",
            json={"row_id": row_id, "grade": grade, "session": session},
        )

    async def ungrade(self, name: str, col: str, row_id: str) -> Dict[str, Any]:
        return await self._request(
            "DELETE", f"/api/exams/{_p(name)}/questions/{_p(col)}/grade/{_p(row_id)}")

    async def suggest(self, name: str, col: str, row_id: str) -> Dict[str, Any]:
        return await self._request(
            "POST", f"/api/exams/{_p(name)}/questions/{_p(col)}/suggest",
            json={"row_id": row_id},
        )

    async def auto_match(self, name: str, col: str, language: Optional[str] = None,
                         top_k: Optional[int] = None) -> Dict[str, Any]:
        return await self._request(
            "POST", f"/api/exams/{_p(name)}/questions/{_p(col)}/auto-match",
            json={"language": language, "top_k": top_k},
        )

    async def question_status(self, name: str, col: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/exams/{_p(name)}/questions/{_p(col)}/status")

    # -- global question bank (app-wide; not exam/course-scoped) ----------
    async def global_bank_status(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/global-bank")

    async def global_bank_reindex(self) -> Dict[str, Any]:
        return await self._request("POST", "/api/global-bank/reindex")

    async def global_bank_import(self, file: str,
                                 bank: Optional[str] = None) -> Dict[str, Any]:
        return await self._request("POST", "/api/global-bank/import",
                                   json={"file": file, "bank": bank})

    async def global_bank_search(self, query: str, language: Optional[str] = None,
                                 top_k: Optional[int] = None) -> Dict[str, Any]:
        return await self._request("POST", "/api/global-bank/search", json={
            "query": query, "language": language, "top_k": top_k,
        })

    async def export(self, name: str) -> Dict[str, str]:
        return await self._download("POST", f"/api/exams/{_p(name)}/export")

    async def export_daisy(self, name: str) -> Dict[str, str]:
        return await self._download("POST", f"/api/exams/{_p(name)}/export/daisy")

    async def export_csv(self, name: str) -> Dict[str, str]:
        return await self._download("POST", f"/api/exams/{_p(name)}/export/csv")

    async def export_results_pdf(self, name: str,
                                 scanned_pdf: Optional[str] = None) -> Dict[str, Any]:
        return await self._request(
            "POST", f"/api/exams/{_p(name)}/export/results-pdf",
            json={"scanned_pdf": scanned_pdf},
        )

    async def download_graded(self, filename: str) -> Dict[str, str]:
        return await self._download("GET", f"/api/graded/{_p(filename)}")

    async def get_results(self, name: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/exams/{_p(name)}/results")

    async def preview_results(self, name: str, scheme: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", f"/api/exams/{_p(name)}/results", json=scheme)

    async def render_data(self, name: str) -> Dict[str, Any]:
        return await self._request("GET", f"/api/exams/{_p(name)}/render-data")

    async def list_exam_scans(self, name: str) -> List[Dict[str, Any]]:
        return await self._request("GET", f"/api/exams/{_p(name)}/scans")

    # -- scheme & question config -----------------------------------------
    # The readable scheme DSL grammar lives in the backend; the TUI round-trips
    # through these rather than parsing/emitting it locally.
    async def scheme_parse(self, text: str) -> Dict[str, Any]:
        return await self._request("POST", "/api/scheme/parse", json={"text": text})

    async def scheme_emit(self, scheme: Dict[str, Any]) -> str:
        data = await self._request("POST", "/api/scheme/emit", json=scheme)
        return data.get("text", "")

    async def put_scheme(self, name: str, scheme: Dict[str, Any]) -> None:
        await self._request("PUT", f"/api/exams/{_p(name)}/scheme", json=scheme)

    async def put_questions_config(self, name: str,
                                   updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self._request(
            "PUT", f"/api/exams/{_p(name)}/questions-config", json=updates)

    # -- import & conflict resolution -------------------------------------
    async def import_preview(self, name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", f"/api/exams/{_p(name)}/import/preview", json=body)

    async def import_apply(self, name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", f"/api/exams/{_p(name)}/import/apply", json=body)

    async def get_conflicts(self, name: str) -> List[Dict[str, Any]]:
        return await self._request("GET", f"/api/exams/{_p(name)}/conflicts")

    async def resolve_conflict(self, name: str, body: Dict[str, Any]) -> None:
        await self._request("POST", f"/api/exams/{_p(name)}/conflicts/resolve", json=body)

    async def upload_file(self, kind: str, path: str) -> Dict[str, Any]:
        """Upload a local file (raw body) into exams/ or scans/."""
        with open(path, "rb") as fh:
            data = fh.read()
        fname = os.path.basename(path)
        resp = await self._send("PUT", f"/api/files/{_p(kind)}/{_p(fname)}", content=data)
        return resp.json()
