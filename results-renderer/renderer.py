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
from collections import OrderedDict

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

    groups = OrderedDict()
    for q in st["questions"]:
        groups.setdefault(q.get("group") or "", []).append(q)

    for gname, qs in groups.items():
        if gname:
            lines.append(rf"\textbf{{Section: {latex_escape(gname)}}}\par\medskip")
        short, essays = [], []
        for q in qs:
            resp = str(q.get("response", "") or "")
            (essays if (q.get("qtype") == "essay" or len(resp) > 120) else short).append(q)

        if short:
            lines.append(r"\renewcommand{\arraystretch}{1.15}")
            lines.append(r"\begin{longtable}{|l|p{0.62\textwidth}|c|}\hline")
            for q in short:
                resp = str(q.get("response", "") or "")
                cell = latex_escape(resp) if resp.strip() not in ("", "-") else r"\textit{(no answer)}"
                mx = q.get("max") or 0
                lines.append(
                    rf"{latex_escape(q.get('label',''))} & {cell} & {_fmt_pts(q.get('points'))} / {mx:.0f} \\ \hline"
                )
            lines.append(r"\end{longtable}")

        for q in essays:
            mx = q.get("max") or 0
            est = " (estimated)" if q.get("estimated") else ""
            lines.append(
                rf"\par\medskip\textbf{{{latex_escape(q.get('label',''))}}} \hfill "
                rf"\textbf{{{_fmt_pts(q.get('points'))}{est}}} / {mx:.0f}"
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
