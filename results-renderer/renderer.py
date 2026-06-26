"""Render a results PDF for an examination.

For each student: a LaTeX answer sheet (responses + marks, grouped by section,
with a Code128 barcode of the id) and, if a scanned exam PDF is provided, the
student's original cover page prepended (matched by PDF417 barcode). All students
are concatenated into one PDF.

Generalized from the PVT pipeline (reference/pvt/scripts) - nothing about
sections or question counts is hardcoded; it works off the examination's
render-data: { exam, students: [ { id, grade, total,
questions: [ { label, group, qtype, response, points, max, estimated } ] } ] }.
"""
import os
import re
import subprocess
import tempfile

import barcode
import pikepdf
from barcode.writer import ImageWriter


def latex_escape(s):
    if s is None:
        return ""
    # Replace backslash with a sentinel first so the braces in its replacement
    # (\textbackslash{}) are not themselves escaped by the brace rules below.
    s = str(s).replace("\\", "\x00")
    for a, b in [
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
        ("<", r"\textless{}"), (">", r"\textgreater{}"),
    ]:
        s = s.replace(a, b)
    return s.replace("\x00", r"\textbackslash{}")


def make_barcode(value, barcode_dir):
    os.makedirs(barcode_dir, exist_ok=True)
    out = os.path.join(barcode_dir, re.sub(r"[^A-Za-z0-9_-]", "_", str(value)))
    png = out + ".png"
    if not os.path.exists(png):
        barcode.Code128(str(value), writer=ImageWriter()).save(
            out,
            options={"write_text": False, "module_height": 6.0, "module_width": 0.22, "quiet_zone": 2},
        )
    return png


def _fmt_pts(p):
    return f"{p:.2f}" if isinstance(p, (int, float)) else "-"




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
    any_est = any(q.get("estimated") for q in st["questions"])
    est_note = r" \textit{(some grades estimated)}" if any_est else ""
    lines += [
        r"\begin{center}",
        rf"\includegraphics[height=28pt]{{{bc}}} \\[6pt]",
        rf"\textbf{{Grade:}} {{\Large\bfseries {latex_escape(st['grade'])}}} \quad "
        rf"\textbf{{Total:}} {st['total']:.2f}{est_note}",
        r"\end{center}",
        r"\vspace{4pt}\hrule\vspace{8pt}",
    ]

    # MC / short-answer table (mc, sc, or unset qtype), then essays, then
    # ungraded comments — mirroring the reference PVT sample layout.
    short, essays, comments = [], [], []
    for i, q in enumerate(st["questions"], 1):
        q = dict(q)
        q["_n"] = i
        qt = (q.get("qtype") or "").lower()
        resp = str(q.get("response", "") or "")
        if qt == "comment":
            comments.append(q)
        elif qt == "essay":
            essays.append(q)
        else:
            short.append(q)

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
        n = q["_n"]
        grp = (q.get("group") or "").strip()
        label = f"{grp} essay (Q{n})" if grp else f"Essay (Q{n})"
        mx = q.get("max") or 0
        pts = _fmt_pts(q.get("points"))
        est = " (estimated)" if q.get("estimated") else ""
        lines.append(
            rf"\par\medskip\textbf{{{latex_escape(label)}}} \hfill "
            rf"\textbf{{{pts}{est}}} / {mx:.0f}"
        )
        lines.append(r"\par\smallskip")
        body = str(q.get("response", "") or "")
        if not body or body.strip() == "-":
            lines.append(r"\textit{(no answer)}")
        else:
            for p in re.split(r"\s{4,}|\n+", body):
                p = p.strip()
                if p:
                    lines.append(latex_escape(p) + r"\par")

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
                if m:
                    sid = m.group(m.lastindex) if m.lastindex else r.text.strip()
                    covers.setdefault(sid, i)
                else:
                    covers.setdefault(r.text.strip(), i)
    return covers


_COVER_IDS_CACHE = {}


def cover_ids(scan_path, id_regex=r"^(\d+)-(\d+)$"):
    """Distinct student ids whose cover page can be decoded from the scan.

    Rasterising + barcode decoding is expensive, so results are cached by
    (path, mtime, size): repeated calls for an untouched file are free.
    """
    try:
        stat = os.stat(scan_path)
    except OSError:
        return []
    key = (os.path.abspath(scan_path), stat.st_mtime, stat.st_size, id_regex)
    cached = _COVER_IDS_CACHE.get(key)
    if cached is None:
        cached = sorted(detect_covers(scan_path, id_regex).keys())
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
            pdf = compile_tex(build_tex_for_student(st, bdir), os.path.join(tmp, f"w_{sid}"))
            with pikepdf.open(pdf) as sp:
                out.pages.extend(sp.pages)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.save(out_path)
    return {"path": out_path, "students": len(students), "covers_missing": missing}
