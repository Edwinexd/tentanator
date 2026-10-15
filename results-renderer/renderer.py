"""Render a results PDF for an examination.

For each student: a LaTeX answer sheet (responses + marks, grouped by section,
with a Code128 barcode of the id) and, if a scanned exam PDF is provided, the
student's original cover page prepended (matched by PDF417 barcode). All students
are concatenated into one PDF.

Generalized from the PVT pipeline (reference/pvt/scripts) - nothing about
sections or question counts is hardcoded; it works off the examination's
render-data: { exam, students: [ { id, grade, rounded_from, total,
questions: [ { label, group, qtype, response, points, max, estimated } ] } ] }.
"""
import json
import os
import re
import subprocess
import tempfile
import threading
import unicodedata

import barcode
import pikepdf
from barcode.writer import ImageWriter


# Non-Latin characters pdflatex (utf8 + T1) cannot print, mapped to LaTeX.
_LATEX_SYMBOLS = {
    "→": r"\ensuremath{\rightarrow}", "➡": r"\ensuremath{\rightarrow}",
    "⟶": r"\ensuremath{\longrightarrow}", "←": r"\ensuremath{\leftarrow}",
    "↔": r"\ensuremath{\leftrightarrow}", "⇒": r"\ensuremath{\Rightarrow}",
    "⇐": r"\ensuremath{\Leftarrow}", "⇔": r"\ensuremath{\Leftrightarrow}",
    "↑": r"\ensuremath{\uparrow}", "↓": r"\ensuremath{\downarrow}",
    "≤": r"\ensuremath{\leq}", "≥": r"\ensuremath{\geq}", "≠": r"\ensuremath{\neq}",
    "≈": r"\ensuremath{\approx}", "∞": r"\ensuremath{\infty}", "−": "-",
    "•": r"\textbullet{}", "…": r"\ldots{}", "€": r"\texteuro{}", "™": r"\texttrademark{}",
}
# Beyond Latin-1/Extended-A, characters utf8 + T1 prints as-is.
_LATEX_OK = set("–—‘’‚“”„†‡‰‹›")


def _latex_char(c):
    """One character pdflatex can print: as-is, a LaTeX symbol, dropped if
    invisible (variation selectors, zero-width), else a visible U+XXXX tag."""
    if ord(c) < 0x180 or c in _LATEX_OK:
        return c
    if c in _LATEX_SYMBOLS:
        return _LATEX_SYMBOLS[c]
    if unicodedata.category(c) in ("Cf", "Mn", "Me") or 0xFE00 <= ord(c) <= 0xFE0F:
        return ""
    return f"[U+{ord(c):04X}]"


def latex_escape(s):
    if s is None:
        return ""
    # Compose decomposed accents (macOS text has "a" + U+0308 for "ä"), which
    # pdflatex rejects; marks with no precomposed form drop in _latex_char.
    s = unicodedata.normalize("NFC", str(s))
    # Replace backslash with a sentinel first so the braces in its replacement
    # (\textbackslash{}) are not themselves escaped by the brace rules below.
    s = s.replace("\\", "\x00")
    for a, b in [
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
        ("<", r"\textless{}"), (">", r"\textgreater{}"),
    ]:
        s = s.replace(a, b)
    s = s.replace("\x00", r"\textbackslash{}")
    return "".join(_latex_char(c) for c in s)


def _code_break(run, prev):
    """Decode one flattened line break: (blank line before?, indent). Moodle's
    export turns each break into 6 spaces plus the next line's indent, and the
    editor auto-indents empty lines, so a blank line is 6 + its own indent + 6
    + the next indent. Python indents by 4 and deepens one level at a time."""
    def ok(indent):
        return indent >= 0 and indent % 4 == 0 and indent <= prev + 4
    if ok(run - 6):
        return False, run - 6
    # Blank line indented like the next line (the editor's usual), else like
    # the previous line, else not at all.
    for blank_indent in ((run - 12) / 2, prev, 0):
        indent = run - 12 - blank_indent
        if indent == int(indent) and ok(int(indent)):
            return True, int(indent)
    return False, run - 6


def code_lines(text):
    """Source lines of a code answer, undoing Moodle's flattened line breaks
    (runs of 6+ spaces; see _code_break)."""
    if "\n" in text:
        return text.rstrip().splitlines()
    parts = re.split(r"( {6,})", text.rstrip())
    lines, line, prev = [], parts[0], 0
    for sep, chunk in zip(parts[1::2], parts[2::2]):
        lines.append(line)
        blank, indent = _code_break(len(sep), prev)
        if blank:
            lines.append("")
        line, prev = " " * indent + chunk, indent
    lines.append(line)
    return lines


def code_block(text):
    """Monospace LaTeX for a code answer, indentation kept, long lines wrap."""
    out = [r"{\ttfamily\small\setlength{\parskip}{0pt}"]
    for line in code_lines(text):
        esc = latex_escape(line).replace(" ", r"\ ")
        out.append((esc or r"\mbox{}") + r"\par")
    out.append("}")
    return out


def answer_block(q):
    """LaTeX for one free-text answer (essay, code or untyped): heading with
    points, then the response (monospace for code)."""
    grp = (q.get("group") or "").strip()
    kind = {"essay": "essay", "code": "code"}.get((q.get("qtype") or "").lower(), "")
    label = " ".join(w for w in (grp, kind) if w) or "Answer"
    label = f"{label[0].upper()}{label[1:]} (Q{q['_n']})"
    est = " (estimated)" if q.get("estimated") else ""
    out = [
        rf"\par\medskip\textbf{{{latex_escape(label)}}} \hfill "
        rf"\textbf{{{_fmt_pts(q.get('points'))}{est}}} / {q.get('max') or 0:.0f}",
        r"\par\smallskip",
    ]
    body = str(q.get("response", "") or "")
    if not body or body.strip() == "-":
        out.append(r"\textit{(no answer)}")
    elif kind == "code":
        out += code_block(body)
    else:
        out += [latex_escape(p.strip()) + r"\par" for p in re.split(r"\s{4,}|\n+", body) if p.strip()]
    return out


def make_barcode(value, barcode_dir):
    os.makedirs(barcode_dir, exist_ok=True)
    out = os.path.join(barcode_dir, _safe_name(value))
    png = out + ".png"
    if not os.path.exists(png):
        barcode.Code128(str(value), writer=ImageWriter()).save(
            out,
            options={"write_text": False, "module_height": 6.0, "module_width": 0.22, "quiet_zone": 2},
        )
    return png


def _safe_name(value):
    """Sanitise a value for use as a single path segment (no traversal)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def _fmt_pts(p):
    return f"{p:.2f}" if isinstance(p, (int, float)) else "-"




def grade_notes(st: dict) -> tuple[str, str]:
    """LaTeX notes after the grade (rounded up) and the total (estimated)."""
    rounded = st.get("rounded_from")
    round_note = rf" \textit{{(rounded up from {latex_escape(rounded)})}}" if rounded else ""
    any_est = any(q.get("estimated") for q in st["questions"])
    est_note = r" \textit{(some grades estimated)}" if any_est else ""
    return round_note, est_note


def build_tex_for_student(st, barcode_dir):
    bc = make_barcode(st["id"], barcode_dir).replace("\\", "/")
    lines = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[english]{babel}",
        r"\usepackage[a4paper,margin=2cm,headheight=22pt,headsep=10pt]{geometry}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{graphicx}",
        r"\usepackage{fancyhdr}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{4pt}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\fancyhead[R]{{\includegraphics[height=12pt]{{{bc}}}}}",
        r"\renewcommand{\headrulewidth}{0pt}",
        r"\fancyfoot[C]{\thepage}",
        r"\begin{document}",
        r"\thispagestyle{fancy}",
    ]
    round_note, est_note = grade_notes(st)
    lines += [
        r"\begin{center}",
        rf"\includegraphics[height=28pt]{{{bc}}} \\[6pt]",
        rf"\textbf{{Grade:}} {{\Large\bfseries {latex_escape(st['grade'])}}}{round_note} \quad "
        rf"\textbf{{Total:}} {st['total']:.2f}{est_note}",
        r"\end{center}",
        r"\vspace{4pt}\hrule\vspace{8pt}",
    ]

    # MC table (mc, sc), then free-text blocks (essay, code, untyped), then
    # ungraded comments. Mirrors the reference PVT sample layout.
    short, essays, comments = [], [], []
    for i, q in enumerate(st["questions"], 1):
        q = dict(q)
        q["_n"] = i
        qt = (q.get("qtype") or "").lower()
        resp = str(q.get("response", "") or "")
        if qt == "comment":
            comments.append(q)
        elif qt in ("mc", "sc"):
            short.append(q)
        else:
            essays.append(q)

    if short:
        lines.append(r"\textbf{Multiple-choice} \hfill {\normalfont\itshape Q | Response | Pts}\par\medskip")
        lines.append(r"\renewcommand{\arraystretch}{1.15}")
        lines.append(r"\begin{longtable}{|c|p{0.66\textwidth}|c|}\hline")
        for q in short:
            resp = str(q.get("response", "") or "")
            cell = latex_escape(resp) if resp.strip() not in ("", "-") else r"\textit{(no answer)}"
            mx = q.get("max") or 0
            lines.append(rf"{q['_n']} & {cell} & {_fmt_pts(q.get('points'))} / {mx:.0f} \\ \hline")
        lines.append(r"\end{longtable}")

    for q in essays:
        lines += answer_block(q)

    for q in comments:
        n = q["_n"]
        body = str(q.get("response", "") or "")
        if not body or body.strip() in ("", "-"):
            continue
        lines.append(rf"\par\medskip\textbf{{Student notes (Q{n}, not graded)}}\par")
        for p in re.split(r"\s{4,}|\n+", body):
            p = p.strip()
            if p:
                lines.append(rf"\textit{{{latex_escape(p)}}}\par")


    lines.append(r"\end{document}")
    return "\n".join(lines)


def compile_tex(tex, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "doc.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
        cwd=work_dir,
        capture_output=True,
    )
    pdf = os.path.join(work_dir, "doc.pdf")
    if not os.path.exists(pdf):
        raise RuntimeError(f"pdflatex failed in {work_dir}")
    return pdf


def detect_covers(pdf_path, id_regex=r"^(\d+)-(\d+)$"):
    """Map student id -> cover page index by decoding barcodes on each page."""
    try:
        import zxingcpp
        from PIL import Image
    except ImportError:
        return {}
    covers = {}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "100", pdf_path, os.path.join(tmp, "p"), "-jpeg"], check=True)
        files = sorted(f for f in os.listdir(tmp) if f.endswith(".jpg"))
        pat = re.compile(id_regex)
        for i, fn in enumerate(files):
            for r in zxingcpp.read_barcodes(Image.open(os.path.join(tmp, fn))):
                m = pat.match(r.text)
                if not m:
                    # Not a student-id barcode (logo QR, page tracker, garbled
                    # decode): ignore it. Counting it as a cover would inflate the
                    # count that gates cover-eligibility and mis-match real students.
                    continue
                sid = m.group(m.lastindex) if m.lastindex else r.text.strip()
                covers.setdefault(sid, i)
    return covers


_COVER_IDS_CACHE = {}
_COVER_IDS_CACHE_MAX = 128
# On-disk copy of the cache, one file per scan directory, so a renderer restart
# (e.g. a rebuild) does not re-decode every scan.
_DISK_CACHE_NAME = ".covers_cache.json"
_DISK_LOCK = threading.Lock()


def _disk_cache_path(scan_path):
    return os.path.join(os.path.dirname(os.path.abspath(scan_path)), _DISK_CACHE_NAME)


def _disk_key(scan_path, stat, id_regex):
    return f"{os.path.basename(scan_path)}|{stat.st_mtime_ns}|{stat.st_size}|{id_regex}"


def _read_disk_cache(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _store_disk_cache(scan_path, key, ids):
    """Merge one entry into the directory's cache file, replacing older entries
    for the same scan and dropping scans that are gone. Best effort: a read-only
    data dir just skips it."""
    path = _disk_cache_path(scan_path)
    folder = os.path.dirname(path)
    with _DISK_LOCK:
        data = _read_disk_cache(path)
        name = os.path.basename(scan_path)
        data = {k: v for k, v in data.items()
                if (n := k.split("|", 1)[0]) != name and os.path.exists(os.path.join(folder, n))}
        data[key] = ids
        try:
            with tempfile.NamedTemporaryFile(
                    "w", dir=folder, prefix=_DISK_CACHE_NAME, suffix=".tmp",
                    delete=False, encoding="utf-8") as tmp:
                json.dump(data, tmp)
            os.replace(tmp.name, path)
        except OSError:
            pass


def cover_ids(scan_path, id_regex=r"^(\d+)-(\d+)$"):
    """Distinct student ids whose cover page can be decoded from the scan.

    Rasterising + barcode decoding is expensive, so results are cached by
    (path, mtime, size): repeated calls for an untouched file are free. The
    in-memory cache is bounded to _COVER_IDS_CACHE_MAX entries, evicting the
    oldest inserted key (FIFO via dict insertion order) when it would overflow,
    and backed by a `.covers_cache.json` next to the scans that survives
    restarts.
    """
    try:
        stat = os.stat(scan_path)
    except OSError:
        return []
    key = (os.path.abspath(scan_path), stat.st_mtime, stat.st_size, id_regex)
    cached = _COVER_IDS_CACHE.get(key)
    if cached is None:
        dkey = _disk_key(scan_path, stat, id_regex)
        cached = _read_disk_cache(_disk_cache_path(scan_path)).get(dkey)
        if not isinstance(cached, list):
            cached = sorted(detect_covers(scan_path, id_regex).keys())
            # An empty decode may be a missing decoder; don't persist that.
            if cached:
                _store_disk_cache(scan_path, dkey, cached)
        if len(_COVER_IDS_CACHE) >= _COVER_IDS_CACHE_MAX:
            oldest = next(iter(_COVER_IDS_CACHE))
            del _COVER_IDS_CACHE[oldest]
        _COVER_IDS_CACHE[key] = cached
    return cached


def render_results(render_data, scanned_pdf_path, out_path, id_regex=r"^(\d+)-(\d+)$"):
    students = render_data.get("students", [])
    covers, original = {}, None
    if scanned_pdf_path and os.path.exists(scanned_pdf_path):
        covers = detect_covers(scanned_pdf_path, id_regex)
        original = pikepdf.open(scanned_pdf_path)

    out = pikepdf.Pdf.new()
    missing = []
    with tempfile.TemporaryDirectory() as tmp:
        bdir = os.path.join(tmp, "barcodes")
        for st in sorted(students, key=lambda s: str(s.get("id", ""))):
            sid = str(st.get("id", ""))
            if original is not None and sid in covers:
                out.pages.append(original.pages[covers[sid]])
            elif original is not None:
                missing.append(sid)
            # sid is sanitised for the work-dir name only; it must not contain
            # path separators or ".." or it could escape the temp dir. The raw
            # sid is still used as the barcode payload (build_tex_for_student).
            pdf = compile_tex(
                build_tex_for_student(st, bdir),
                os.path.join(tmp, f"w_{_safe_name(sid)}"),
            )
            with pikepdf.open(pdf) as sp:
                out.pages.extend(sp.pages)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path)
    return {"path": out_path, "students": len(students), "covers_missing": missing}
