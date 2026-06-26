"""FastAPI wrapper around the results renderer.

POST /render { exam, scanned_pdf? } fetches the examination's render-data from
the backend, reads the (optional) scanned exam PDF from the shared data volume,
and writes a concatenated results PDF into graded_exams/.
"""
import os
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from renderer import cover_ids, render_results

API = os.getenv("TENTANATOR_API", "http://backend:8787")
DATA = os.getenv("TENTANATOR_DATA_DIR", "/data")

app = FastAPI(title="Tentanator results renderer")


class RenderReq(BaseModel):
    exam: str
    scanned_pdf: Optional[str] = None


class CoversReq(BaseModel):
    scanned_pdf: str


def _resolve_scan(scanned_pdf: str) -> str:
    fname = os.path.basename(scanned_pdf)  # no path traversal
    for sub in ("scans", "exams", ""):
        cand = os.path.join(DATA, sub, fname)
        if os.path.exists(cand):
            return cand
    raise HTTPException(404, f"scanned PDF not found: {scanned_pdf}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/covers")
def covers(req: CoversReq):
    """Count the distinct cover pages (barcodes) decodable from a scanned PDF.

    Used by the backend to offer only scans whose front-page count matches the
    examination's student count.
    """
    scan = _resolve_scan(req.scanned_pdf)
    ids = cover_ids(scan)
    return {"filename": os.path.basename(scan), "count": len(ids), "ids": ids}


@app.post("/render")
def render(req: RenderReq):
    try:
        r = httpx.get(f"{API}/api/exams/{req.exam}/render-data", timeout=180)
    except httpx.RequestError as e:
        raise HTTPException(502, f"cannot reach backend: {e}")
    if r.status_code != 200:
        raise HTTPException(502, f"render-data fetch failed: {r.text}")
    data = r.json()

    scanned = _resolve_scan(req.scanned_pdf) if req.scanned_pdf else None

    out_path = os.path.join(DATA, "graded_exams", f"{req.exam}_results.pdf")
    try:
        return render_results(data, scanned, out_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise HTTPException(500, f"render failed: {e}")
