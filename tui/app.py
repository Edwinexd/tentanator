"""Tentanator Textual TUI.

A terminal client over the Rust backend HTTP API. It contains presentation +
navigation only; grading, sampling, persistence and LLM calls all live in the
backend.

Run:
    TENTANATOR_API=http://127.0.0.1:8787 python app.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
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


def row_id(row: Dict[str, str], id_columns: List[str]) -> str:
    """Mirror the backend's row id: ID column values joined by '_'."""
    return "_".join(row.get(c, "") for c in id_columns)


def is_meaningful(text: str) -> bool:
    t = (text or "").strip()
    return t not in ("", "-", "N/A")


class SessionListScreen(Screen):
    """Landing screen: pick an existing session or start a new one."""

    BINDINGS = [("n", "new_session", "New session"), ("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("Sessions", id="title")
        yield ListView(id="sessions")
        with Horizontal(id="actions"):
            yield Button("New session [n]", id="new", variant="primary")
            yield Button("Refresh [r]", id="refresh")
        yield Footer()

    async def on_mount(self) -> None:
        await self.refresh_sessions()

    async def refresh_sessions(self) -> None:
        lv = self.query_one("#sessions", ListView)
        await lv.clear()
        try:
            sessions = await self.app.api.list_sessions()  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        if not sessions:
            await lv.append(ListItem(Label("(no sessions - press 'n' to create one)")))
            return
        for s in sessions:
            label = (
                f"{s['session_name']}  -  {s['csv_file']}  "
                f"({s['num_questions']} q, updated {s['last_updated'][:19]})"
            )
            await lv.append(ListItem(Label(label), name=s["session_name"]))

    def action_new_session(self) -> None:
        self.app.push_screen(NewSessionScreen())

    async def action_refresh(self) -> None:
        await self.refresh_sessions()

    @on(Button.Pressed, "#new")
    def _new(self) -> None:
        self.action_new_session()

    @on(Button.Pressed, "#refresh")
    async def _refresh(self) -> None:
        await self.refresh_sessions()

    @on(ListView.Selected)
    def _open(self, event: ListView.Selected) -> None:
        if event.item is not None and event.item.name:
            self.app.push_screen(GradingScreen(event.item.name))


class NewSessionScreen(Screen):
    """Wizard: choose an exam file then the id / input / output columns."""

    BINDINGS = [("escape", "cancel", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._columns: List[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label("New session", id="title")
            yield Label("Exam file:")
            yield Select([], id="examfile", prompt="Select an exam file")
            yield Label("ID columns (student identifier):")
            yield SelectionList(id="idcols")
            yield Label("Input columns (student responses):")
            yield SelectionList(id="inputcols")
            yield Label("Output columns (one per graded question):")
            yield SelectionList(id="outputcols")
            yield Label("Session name (optional):")
            yield Input(placeholder="auto-generated if blank", id="sessionname")
            with Horizontal(id="actions"):
                yield Button("Create", id="create", variant="primary")
                yield Button("Cancel [esc]", id="cancel")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            exams = await self.app.api.list_exams()  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.query_one("#examfile", Select).set_options([(e, e) for e in exams])

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
        name = self.query_one("#sessionname", Input).value.strip()
        payload: Dict[str, Any] = {
            "csv_file": str(examfile),
            "id_columns": id_cols,
            "input_columns": input_cols,
            "output_columns": output_cols,
        }
        if name:
            payload["name"] = name
        try:
            session = await self.app.api.create_session(payload)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.app.pop_screen()
        self.app.push_screen(GradingScreen(session["session_name"]))


class GradingScreen(Screen):
    """Grade one output column at a time, with optional AI suggestions."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("a", "suggest", "AI suggest"),
        ("s", "skip", "Skip"),
        ("e", "export", "Export"),
    ]

    def __init__(self, session_name: str) -> None:
        super().__init__()
        self.session_name = session_name
        self.session: Dict[str, Any] = {}
        self.rows: List[Dict[str, str]] = []
        self.id_columns: List[str] = []
        self.current_col: Optional[str] = None
        self.ungraded: List[Dict[str, str]] = []
        self.index: int = 0
        self.suggestion: Optional[Dict[str, Any]] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="topbar"):
            yield Select([], id="colselect", prompt="Question")
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
            self.session = await self.app.api.get_session(self.session_name)  # type: ignore[attr-defined]
            self.rows = await self.app.api.exam_rows(self.session["csv_file"])  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.id_columns = self.session.get("id_columns", [])
        out_cols = self.session.get("output_columns", [])
        sel = self.query_one("#colselect", Select)
        sel.set_options([(c, c) for c in out_cols])
        if out_cols:
            sel.value = out_cols[0]
            self.current_col = out_cols[0]
            await self.refresh_question()

    async def refresh_question(self) -> None:
        if not self.current_col:
            return
        question = self.session.get("questions", {}).get(self.current_col, {})
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
        question = self.session.get("questions", {}).get(self.current_col, {})
        input_col = question.get("input_column", "")
        total = len(self.rows)
        graded = len(question.get("graded_items", []))
        try:
            status = await self.app.api.question_status(self.session_name, self.current_col)  # type: ignore[attr-defined]
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
            self.suggestion = await self.app.api.suggest(self.session_name, self.current_col, rid)  # type: ignore[attr-defined]
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
            question = await self.app.api.grade(self.session_name, self.current_col, rid, grade)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.session.setdefault("questions", {})[self.current_col] = question
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
            result = await self.app.api.sampling(self.session_name, self.current_col, algorithm)  # type: ignore[attr-defined]
            self.session = await self.app.api.get_session(self.session_name)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(f"Selected {result.get('num_samples', 0)} representative samples")
        await self.refresh_question()

    @on(Button.Pressed, "#export")
    async def action_export(self) -> None:
        try:
            result = await self.app.api.export(self.session_name)  # type: ignore[attr-defined]
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(f"Exported to {result.get('path', '?')}", timeout=8)

    def action_back(self) -> None:
        self.app.pop_screen()


class TentanatorTUI(App):
    """Top-level app holding the shared API client."""

    CSS = """
    #title { padding: 1 2; text-style: bold; }
    #sessions { height: 1fr; border: round $primary; margin: 0 1; }
    #actions { height: auto; padding: 1 2; }
    #topbar { height: auto; padding: 1 1; }
    #topbar Select { width: 40; }
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
        self.push_screen(SessionListScreen())

    async def on_unmount(self) -> None:
        await self.api.aclose()


def main() -> None:
    TentanatorTUI().run()


if __name__ == "__main__":
    main()
