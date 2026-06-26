"""Tentanator Textual TUI.

A terminal client over the Rust backend HTTP API. It contains presentation +
navigation only; grading, sampling, persistence and LLM calls all live in the
backend.

Run:
    TENTANATOR_API=http://127.0.0.1:8787 python app.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    SelectionList,
    Static,
)

from api import APIError, TentanatorAPI


class FilePickerScreen(Screen):
    """Browse the local filesystem and upload a file into exams/ or scans/."""

    BINDINGS = [("escape", "cancel", "Back")]

    def __init__(self, default_kind: str = "exams") -> None:
        super().__init__()
        self.default_kind = default_kind

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Pick a file to upload (Enter on a file)")
        with Horizontal(id="fpbar"):
            yield Label("Upload as: ")
            yield Select(
                [("exam / grades file (exams/)", "exams"), ("scanned exam PDF (scans/)", "scans")],
                value=self.default_kind,
                allow_blank=False,
                id="fpkind",
            )
        yield DirectoryTree(str(Path.home()), id="fptree")
        yield Footer()

    @on(DirectoryTree.FileSelected)
    async def _selected(self, event: DirectoryTree.FileSelected) -> None:
        kind = str(self.query_one("#fpkind", Select).value)
        path = str(event.path)
        self.notify(f"Uploading {Path(path).name}…", timeout=3)
        try:
            res = await self.app.api.upload_file(kind, path)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        except OSError as exc:
            self.notify(f"Cannot read file: {exc}", severity="error", timeout=8)
            return
        self.notify(f"Uploaded {res.get('filename')} to {kind}/")
        self.app.pop_screen()

    def action_cancel(self) -> None:
        self.app.pop_screen()


def row_id(row: Dict[str, str], id_columns: List[str]) -> str:
    """Mirror the backend's row id: ID column values joined by '_'."""
    return "_".join(row.get(c, "") for c in id_columns)


def is_meaningful(text: str) -> bool:
    t = (text or "").strip()
    return t not in ("", "-", "N/A")


class ExamListScreen(Screen):
    """Landing screen: pick an existing exam or start a new one."""

    BINDINGS = [
        ("n", "new_exam", "New exam"),
        ("u", "upload", "Upload file"),
        ("i", "import_legacy", "Import legacy"),
        ("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("Exams", id="title")
        yield ListView(id="exams")
        with Horizontal(id="actions"):
            yield Button("New exam [n]", id="new", variant="primary")
            yield Button("Refresh [r]", id="refresh")
        yield Footer()

    async def on_mount(self) -> None:
        await self.refresh_exams()

    async def refresh_exams(self) -> None:
        lv = self.query_one("#exams", ListView)
        await lv.clear()
        try:
            exams = await self.app.api.list_exams()  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        if not exams:
            await lv.append(ListItem(Label("(no exams - press 'n' to create one)")))
            return
        for e in exams:
            course = f"  [{e['course']}]" if e.get("course") else ""
            label = (
                f"{e['name']}{course}  -  {e['exam_file']}  "
                f"({e['num_questions']} q, updated {e['last_updated'][:19]})"
            )
            await lv.append(ListItem(Label(label), name=e["name"]))

    def action_new_exam(self) -> None:
        self.app.push_screen(NewExamScreen())

    def action_upload(self) -> None:
        self.app.push_screen(FilePickerScreen())

    async def action_import_legacy(self) -> None:
        """Import loose legacy `.tentanator_sessions/` exams into the new format."""
        try:
            result = await self.app.api.import_legacy_sessions()  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        imported = result.get("imported_exams", [])
        self.notify(f"Imported {len(imported)} legacy exam(s)")
        await self.refresh_exams()

    async def action_refresh(self) -> None:
        await self.refresh_exams()

    @on(Button.Pressed, "#new")
    def _new(self) -> None:
        self.action_new_exam()

    @on(Button.Pressed, "#refresh")
    async def _refresh(self) -> None:
        await self.refresh_exams()

    @on(ListView.Selected)
    def _open(self, event: ListView.Selected) -> None:
        if event.item is not None and event.item.name:
            self.app.push_screen(GradingScreen(event.item.name))


class NewExamScreen(Screen):
    """Wizard: choose an exam file then the id / input / output columns."""

    BINDINGS = [("escape", "cancel", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._columns: List[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label("New exam", id="title")
            yield Label("Exam file:")
            yield Select([], id="examfile", prompt="Select an exam file")
            yield Label("ID columns (student identifier):")
            yield SelectionList(id="idcols")
            yield Label("Input columns (student responses):")
            yield SelectionList(id="inputcols")
            yield Label("Output columns (one per graded question):")
            yield SelectionList(id="outputcols")
            yield Label("Course (optional):")
            yield Input(placeholder="e.g. CS101", id="course")
            yield Label("Exam name (optional):")
            yield Input(placeholder="auto-generated if blank", id="examname")
            with Horizontal(id="actions"):
                yield Button("Create", id="create", variant="primary")
                yield Button("Cancel [esc]", id="cancel")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            files = await self.app.api.list_exam_files()  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.query_one("#examfile", Select).set_options([(e, e) for e in files])

    @on(Select.Changed, "#examfile")
    async def _exam_chosen(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        try:
            cols = await self.app.api.exam_columns(str(event.value))  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self._columns = cols
        options = [(c, c) for c in cols]
        for sel_id in ("#idcols", "#inputcols", "#outputcols"):
            sl = self.query_one(sel_id, SelectionList)
            sl.clear_options()
            sl.add_options(options)

    def action_cancel(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.action_cancel()

    @on(Button.Pressed, "#create")
    async def _create(self) -> None:
        examfile = self.query_one("#examfile", Select).value
        if examfile is Select.BLANK:
            self.notify("Pick an exam file first", severity="warning")
            return
        id_cols = list(self.query_one("#idcols", SelectionList).selected)
        input_cols = list(self.query_one("#inputcols", SelectionList).selected)
        output_cols = list(self.query_one("#outputcols", SelectionList).selected)
        if not input_cols or not output_cols:
            self.notify("Select at least one input and one output column", severity="warning")
            return
        name = self.query_one("#examname", Input).value.strip()
        course = self.query_one("#course", Input).value.strip()
        payload: Dict[str, Any] = {
            "exam_file": str(examfile),
            "id_columns": id_cols,
            "input_columns": input_cols,
            "output_columns": output_cols,
        }
        if name:
            payload["name"] = name
        if course:
            payload["course"] = course
        try:
            exam = await self.app.api.create_exam(payload)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.app.pop_screen()
        self.app.push_screen(GradingScreen(exam["name"]))


class GradingScreen(Screen):
    """Grade one output column at a time, with optional AI suggestions."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("a", "suggest", "AI suggest"),
        ("s", "skip", "Skip"),
        ("e", "export", "Export"),
        ("r", "results", "Results"),
    ]

    def __init__(self, exam_name: str) -> None:
        super().__init__()
        self.exam_name = exam_name
        self.exam: Dict[str, Any] = {}
        self.rows: List[Dict[str, str]] = []
        self.id_columns: List[str] = []
        self.current_col: Optional[str] = None
        self.ungraded: List[Dict[str, str]] = []
        self.index: int = 0
        self.suggestion: Optional[Dict[str, Any]] = None
        self.active_session: str = "default"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="topbar"):
            yield Select([], id="colselect", prompt="Question")
            yield Select([], id="sessionselect", prompt="Session", allow_blank=False)
            yield Button("session+", id="newsession")
            yield Button("random", id="samp-random")
            yield Button("maximin", id="samp-maximin")
            yield Button("export [e]", id="export")
        yield Static("", id="progress")
        with VerticalScroll(id="responsebox"):
            yield Static("", id="response")
        yield Static("", id="aibox")
        with Horizontal(id="gradebar"):
            yield Input(placeholder="grade e.g. 7.5 or 2+1.5", id="gradeinput")
            yield Button("AI suggest [a]", id="suggest")
            yield Button("save", id="save", variant="primary")
            yield Button("skip [s]", id="skip")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            self.exam = await self.app.api.get_exam(self.exam_name)  # type: ignore[attr-defined]
            self.rows = await self.app.api.exam_rows(self.exam["exam_file"])  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.id_columns = self.exam.get("id_columns", [])
        await self.refresh_sessions()
        out_cols = self.exam.get("output_columns", [])
        sel = self.query_one("#colselect", Select)
        sel.set_options([(c, c) for c in out_cols])
        if out_cols:
            sel.value = out_cols[0]
            self.current_col = out_cols[0]
            await self.refresh_question()

    async def refresh_sessions(self) -> None:
        """Reload the session list and keep the active session valid."""
        try:
            sessions = await self.app.api.list_sessions(self.exam_name)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        names = [s["name"] for s in sessions]
        sel = self.query_one("#sessionselect", Select)
        sel.set_options([(f"{s['name']} ({s['graded_count']})", s["name"]) for s in sessions])
        if self.active_session not in names:
            self.active_session = "default" if "default" in names else (names[0] if names else "default")
        if names:
            sel.value = self.active_session

    @on(Select.Changed, "#sessionselect")
    def _session_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self.active_session = str(event.value)

    @on(Button.Pressed, "#newsession")
    async def _new_session(self) -> None:
        try:
            created = await self.app.api.create_session(self.exam_name)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.active_session = created["name"]
        await self.refresh_sessions()
        self.notify(f"New session: {created['name']}")

    async def refresh_question(self) -> None:
        if not self.current_col:
            return
        question = self.exam.get("questions", {}).get(self.current_col, {})
        input_col = question.get("input_column", "")
        graded_ids = {gi["row_id"] for gi in question.get("graded_items", [])}
        sampling = question.get("sampling_result") or {}
        priority = list(sampling.get("selected_ids", []))

        candidates = [
            r for r in self.rows
            if row_id(r, self.id_columns) not in graded_ids
            and is_meaningful(r.get(input_col, ""))
        ]
        # Put sampled rows first so representative responses are graded early.
        candidates.sort(key=lambda r: (row_id(r, self.id_columns) not in priority,))
        self.ungraded = candidates
        self.index = 0
        self.suggestion = None
        await self.show_current()

    async def show_current(self) -> None:
        question = self.exam.get("questions", {}).get(self.current_col, {})
        input_col = question.get("input_column", "")
        total = len(self.rows)
        graded = len(question.get("graded_items", []))
        try:
            status = await self.app.api.question_status(self.exam_name, self.current_col)  # type: ignore[attr-defined]
            icl = "yes" if status.get("icl_ready") else "no"
        except APIError:
            icl = "?"
        self.query_one("#aibox", Static).update("")
        self.query_one("#gradeinput", Input).value = ""

        if not self.ungraded:
            self.query_one("#progress", Static).update(
                f"[b]{self.current_col}[/b]  -  all {total} rows handled  (graded {graded})"
            )
            self.query_one("#response", Static).update("Nothing left to grade for this question.")
            return

        row = self.ungraded[self.index]
        rid = row_id(row, self.id_columns)
        text = row.get(input_col, "")
        words = len(text.split())
        self.query_one("#progress", Static).update(
            f"[b]{self.current_col}[/b]  -  {self.index + 1}/{len(self.ungraded)} ungraded  "
            f"(graded {graded}/{total}, ICL ready: {icl})  -  id: {rid}"
        )
        self.query_one("#response", Static).update(f"[dim]({words} words)[/dim]\n\n{text}")

    def _current_row_id(self) -> Optional[str]:
        if not self.ungraded:
            return None
        return row_id(self.ungraded[self.index], self.id_columns)

    @on(Select.Changed, "#colselect")
    async def _col_changed(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        self.current_col = str(event.value)
        await self.refresh_question()

    @on(Button.Pressed, "#suggest")
    async def action_suggest(self) -> None:
        rid = self._current_row_id()
        if not rid or not self.current_col:
            return
        self.query_one("#aibox", Static).update("[dim]Asking the model...[/dim]")
        try:
            self.suggestion = await self.app.api.suggest(self.exam_name, self.current_col, rid)  # type: ignore[attr-defined]
        except APIError as exc:
            self.query_one("#aibox", Static).update("")
            self.notify(str(exc), severity="error", timeout=8)
            return
        grade = self.suggestion.get("grade", "")
        reasoning = self.suggestion.get("reasoning_summary") or ""
        box = f"[b]AI grade:[/b] {grade}"
        if reasoning:
            box += f"\n[dim]{reasoning}[/dim]"
        self.query_one("#aibox", Static).update(box)
        self.query_one("#gradeinput", Input).value = grade

    @on(Button.Pressed, "#save")
    async def _save_button(self) -> None:
        await self._save()

    @on(Input.Submitted, "#gradeinput")
    async def _save_submit(self) -> None:
        await self._save()

    async def _save(self) -> None:
        rid = self._current_row_id()
        if not rid or not self.current_col:
            return
        grade = self.query_one("#gradeinput", Input).value.strip()
        if not grade:
            self.notify("Enter a grade (or press 'a' for an AI suggestion)", severity="warning")
            return
        try:
            question = await self.app.api.grade(  # type: ignore[attr-defined]
                self.exam_name, self.current_col, rid, grade, self.active_session
            )
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.exam.setdefault("questions", {})[self.current_col] = question
        # Drop the just-graded row from the queue and stay on the same index.
        del self.ungraded[self.index]
        if self.index >= len(self.ungraded):
            self.index = max(0, len(self.ungraded) - 1)
        self.suggestion = None
        await self.show_current()

    def action_skip(self) -> None:
        if self.ungraded:
            self.index = (self.index + 1) % len(self.ungraded)
            self.run_worker(self.show_current())

    @on(Button.Pressed, "#skip")
    def _skip_button(self) -> None:
        self.action_skip()

    @on(Button.Pressed, "#samp-random")
    async def _sample_random(self) -> None:
        await self._sample("random")

    @on(Button.Pressed, "#samp-maximin")
    async def _sample_maximin(self) -> None:
        await self._sample("maximin")

    async def _sample(self, algorithm: str) -> None:
        if not self.current_col:
            return
        self.notify(f"Sampling ({algorithm})...", timeout=3)
        try:
            result = await self.app.api.sampling(self.exam_name, self.current_col, algorithm)  # type: ignore[attr-defined]
            self.exam = await self.app.api.get_exam(self.exam_name)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(f"Selected {result.get('num_samples', 0)} representative samples")
        await self.refresh_question()

    @on(Button.Pressed, "#export")
    async def action_export(self) -> None:
        try:
            result = await self.app.api.export(self.exam_name)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(f"Exported to {result.get('path', '?')}", timeout=8)

    async def action_results(self) -> None:
        try:
            data = await self.app.api.get_results(self.exam_name)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        if not data.get("has_scheme"):
            self.notify("No grade scheme set (configure it in the web Scheme tab)", timeout=6)
            return
        dist = data.get("distribution", {})
        dist_str = "  ".join(f"{g}:{c}" for g, c in sorted(dist.items()))
        self.notify(
            f"Fully graded {data.get('complete', 0)}/{data.get('total_students', 0)}"
            f"   |   {dist_str}",
            timeout=10,
        )

    def action_back(self) -> None:
        self.app.pop_screen()


class TentanatorTUI(App):
    """Top-level app holding the shared API client."""

    CSS = """
    #title { padding: 1 2; text-style: bold; }
    #exams { height: 1fr; border: round $primary; margin: 0 1; }
    #actions { height: auto; padding: 1 2; }
    #topbar { height: auto; padding: 1 1; }
    #colselect { width: 32; }
    #sessionselect { width: 24; }
    #progress { padding: 1 2; }
    #responsebox { height: 1fr; border: round $primary; margin: 0 1; padding: 1; }
    #aibox { padding: 1 2; height: auto; }
    #gradebar { height: auto; padding: 1 1; }
    #gradeinput { width: 1fr; }
    Button { margin: 0 1; }
    """

    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def __init__(self, api: Optional[TentanatorAPI] = None) -> None:
        super().__init__()
        self.api = api or TentanatorAPI()

    def on_mount(self) -> None:
        self.push_screen(ExamListScreen())

    async def on_unmount(self) -> None:
        await self.api.aclose()


def main() -> None:
    TentanatorTUI().run()


if __name__ == "__main__":
    main()
