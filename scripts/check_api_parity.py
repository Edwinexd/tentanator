#!/usr/bin/env python3
"""Contract-parity check for the Tentanator clients and docs.

The backend (`backend/src/routes.rs`) is the single source of truth. The TUI
(`tui/api.py`) and Web (`web/src/lib/api.ts`) are thin clients over the same
HTTP API and must each cover every route; the contract tables in
`ARCHITECTURE.md` must list every route too. This script extracts the
`(method, path)` surface from the router, from each client, and from the doc
tables, then fails if any of them is missing a route (or names one the backend
does not expose).

Path parameters are normalised so `/api/exams/{name}` (Rust / docs),
`/api/exams/${enc(name)}` (TS) and `/api/exams/{name}` (Python f-string) all
compare equal. Query strings are ignored.

Intentional, documented exclusions go in ALLOW_MISSING. Run from anywhere:

    python3 scripts/check_api_parity.py        # exit 1 on any gap
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
ROUTES_RS = ROOT / "backend" / "src" / "routes.rs"
WEB_API = ROOT / "web" / "src" / "lib" / "api.ts"
TUI_API = ROOT / "tui" / "api.py"
ARCH_MD = ROOT / "ARCHITECTURE.md"

Route = Tuple[str, str]  # (METHOD, normalised path)

# Routes a consumer is allowed to skip, with a reason. Keep this empty unless an
# omission is genuinely consumer-specific; a gap here is a parity bug, not a TODO.
ALLOW_MISSING: Dict[str, Set[Route]] = {
    "web": set(),
    "tui": set(),
    "docs": set(),
}


def normalize(path: str) -> str:
    """Canonicalise a route path: drop query, collapse params to ':p'."""
    path = path.split("?", 1)[0]
    path = re.sub(r"\$\{[^}]*\}", ":p", path)  # TS  ${enc(name)}
    path = re.sub(r"\{[^}]*\}", ":p", path)    # Rust {name} / Python f"{name}"
    return path.rstrip("/")


def backend_routes() -> Set[Route]:
    """Extract (method, path) pairs from the Axum router in routes.rs.

    Only a single flat ``Router::new()...with_state()`` span is parsed. If
    routes are ever registered outside that span - via ``.nest(...)``,
    ``.merge(...)``, or a second router - they would be invisible to every
    surface and parity would pass while real routes are uncovered (a
    false-pass, the worst failure for a parity gate). Two guards prevent that:
    reject any ``.nest(``/``.merge(`` outright, and assert the number of
    ``.route(`` occurrences in the whole file equals the number parsed here.
    """
    src = ROUTES_RS.read_text(encoding="utf-8")
    for combinator in (".nest(", ".merge("):
        if combinator in src:
            sys.exit(
                f"{ROUTES_RS} uses '{combinator}'; this script only parses a single flat "
                "Router::new()...with_state() span and would silently miss those routes. "
                "Update backend_routes() to handle nested/merged routers before relying on it."
            )
    body = re.search(r"Router::new\(\)(.*?)\.with_state\(", src, re.DOTALL)
    if not body:
        sys.exit(f"could not locate Router::new()..with_state() in {ROUTES_RS}")
    chunks = body.group(1).split(".route(")[1:]
    parsed = len(chunks)
    total = src.count(".route(")
    if parsed != total:
        sys.exit(
            f"{ROUTES_RS} has {total} '.route(' occurrence(s) but only {parsed} fall inside the "
            "parsed Router::new()...with_state() span; routes are registered elsewhere and would "
            "be missed. Update backend_routes() to cover them."
        )
    routes: Set[Route] = set()
    for chunk in chunks:
        path_m = re.search(r'"([^"]+)"', chunk)
        if not path_m:
            continue
        path = normalize(path_m.group(1))
        for verb in re.findall(r"\b(get|post|put|delete|patch)\s*\(", chunk):
            routes.add((verb.upper(), path))
    return routes


def web_routes() -> Set[Route]:
    """Extract (method, path) pairs called by the web client (api.ts)."""
    src = WEB_API.read_text(encoding="utf-8")
    routes: Set[Route] = set()
    # req<...>('GET', `/api/...`) and triggerDownload('POST', `/api/...`)
    for method, path in re.findall(
        r"\b(?:req|triggerDownload)\b[^(]*\(\s*['\"]([A-Z]+)['\"]\s*,\s*[`'\"]([^`'\"]+)",
        src,
    ):
        routes.add((method.upper(), normalize(path)))
    # Raw fetch with a literal path (uploadBinary): method is in the options obj.
    for path, tail in re.findall(
        r"fetch\(\s*`\$\{API_BASE\}(/api[^`]*)`\s*,\s*\{(.*?)\}", src, re.DOTALL
    ):
        verb = re.search(r"method:\s*['\"](\w+)['\"]", tail)
        routes.add(((verb.group(1) if verb else "GET").upper(), normalize(path)))
    return routes


def tui_routes() -> Set[Route]:
    """Extract (method, path) pairs called by the TUI client (api.py)."""
    src = TUI_API.read_text(encoding="utf-8")
    routes: Set[Route] = set()
    for method, path in re.findall(
        r"self\._(?:send|download|request|client\.request)\(\s*\"([A-Z]+)\"\s*,\s*f?\"([^\"]+)\"",
        src,
    ):
        routes.add((method.upper(), normalize(path)))
    return routes


def doc_routes() -> Set[Route]:
    """Extract (method, path) pairs from the contract tables in ARCHITECTURE.md."""
    src = ARCH_MD.read_text(encoding="utf-8")
    routes: Set[Route] = set()
    # Markdown table rows: | METHOD | `/api/...` | ... |. Prose mentions (no
    # leading pipe) are ignored, so only the structured tables are the contract.
    for method, path in re.findall(
        r"^\|\s*(GET|POST|PUT|DELETE|PATCH)\s*\|\s*`([^`]+)`", src, re.MULTILINE
    ):
        routes.add((method.upper(), normalize(path)))
    return routes


def fmt(routes: Set[Route]) -> str:
    """Render a sorted route set for human-readable output."""
    return "\n".join(f"  {m:6} {p}" for m, p in sorted(routes, key=lambda r: (r[1], r[0])))


def main() -> int:
    """Compare each consumer (clients + docs) against the backend; 1 on any gap."""
    backend = backend_routes()
    if not backend:
        sys.exit("parsed zero backend routes - routes.rs format changed?")
    consumers = {"web": web_routes(), "tui": tui_routes(), "docs": doc_routes()}

    print(f"backend exposes {len(backend)} (method, path) routes")
    ok = True
    for name, got in consumers.items():
        if not got:
            print(f"\n{name}: parsed zero routes - source format changed?")
            ok = False
            continue
        missing = backend - got - ALLOW_MISSING[name]
        extra = got - backend - ALLOW_MISSING[name]
        allowed = (backend - got) & ALLOW_MISSING[name]
        status = "OK" if not missing and not extra else "FAIL"
        print(f"\n{name}: covers {len(got & backend)}/{len(backend)}  [{status}]")
        if allowed:
            print(f" allow-listed (skipped on purpose): {len(allowed)}")
        if missing:
            ok = False
            print(f" MISSING {len(missing)} backend route(s):\n{fmt(missing)}")
        if extra:
            ok = False
            print(f" EXTRA {len(extra)} route(s) with no backend match:\n{fmt(extra)}")

    print("\nparity OK" if ok else "\nparity FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
