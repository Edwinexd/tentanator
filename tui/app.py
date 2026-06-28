"""Tentanator Textual TUI.

A terminal client over the Rust backend HTTP API. It contains presentation +
navigation only; grading, sampling, scheme parsing, persistence and LLM calls
all live in the backend. The TUI is an equal of the web client: every task the
web GUI can do is reachable here too.

Run:
    TENTANATOR_API=http://127.0.0.1:8787 python app.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
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
    Switch,
    TextArea,
)
from textual.widgets.selection_list import Selection

from api import APIError, TentanatorAPI

# Event handlers are trivial Textual glue (a button press -> one API call); the
# screen-level docstrings already describe behavior, so per-handler docstrings
# add only noise. GradingScreen tracks the grading cursor across several fields.
# The TUI is one cohesive single-file client, hence the module length.
# pylint: disable=missing-function-docstring,too-many-instance-attributes,too-many-lines

# Presentation helpers -------------------------------------------------------


def row_id(row: Dict[str, str], id_columns: List[str]) -> str:
    """Mirror the backend's row id: ID column values joined by '_'."""
    return "_".join(row.get(c, "") for c in id_columns)


def is_meaningful(text: str) -> bool:
    """Mirror the backend's meaningful-response test (display ordering only)."""
    t = (text or "").strip()
    return t not in ("", "-", "N/A")


def match_question(match: Dict[str, Any], language: Optional[str]) -> str:
    """Pick a GlobalBankMatch's question text for the (detected) language."""
    # Swedish prefers q_se; English (and the default) prefers q_en.
    if language == "se":
        return match.get("q_se") or match.get("q_en") or ""
    return match.get("q_en") or match.get("q_se") or ""


def match_answer(match: Dict[str, Any], language: Optional[str]) -> str:
    """Pick a GlobalBankMatch's sample answer for the (detected) language."""
    # Swedish prefers ans_se; English (and the default) prefers ans_en.
    if language == "se":
        return match.get("ans_se") or match.get("ans_en") or ""
    return match.get("ans_en") or match.get("ans_se") or ""


class TentanatorScreen(Screen):
    """Base screen exposing the shared backend client with a precise type."""

    @property
    def api(self) -> TentanatorAPI:
        return self.app.api  # type: ignore[attr-defined,no-any-return]


# Generic widgets ------------------------------------------------------------


class TextPromptScreen(ModalScreen[Optional[str]]):
    """Modal one-line text prompt; dismisses with the entered value or None."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, prompt: str, initial: str = "") -> None:
        super().__init__()
        self.prompt = prompt
        self.initial = initial

    def compose(self) -> ComposeResult:
        with Vertical(id="promptbox"):
            yield Label(self.prompt)
            yield Input(value=self.initial, id="promptinput")
            with Horizontal(id="promptactions"):
                yield Button("OK", id="ok", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#promptinput", Input).focus()

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted, "#promptinput")
    def _ok(self) -> None:
        self.dismiss(self.query_one("#promptinput", Input).value)

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class FilePickerScreen(TentanatorScreen):
    """Browse the local filesystem and upload a file into exams/, scans/ or raw."""

    BINDINGS = [("escape", "cancel", "Back")]

    _KIND_LABELS = [
        ("exam / grades file (exams/)", "exams"),
        ("scanned exam PDF (scans/)", "scans"),
        ("raw Moodle export (exams_in_raw/)", "raw"),
    ]

    def __init__(
        self, default_kind: str = "exams", kinds: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        self.default_kind = default_kind
        # When ``kinds`` is given, lock the picker to that single upload target
        # (no kind selector); otherwise offer the full set.
        self.kinds = kinds

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Pick a file to upload (Enter on a file)")
        if self.kinds is None:
            with Horizontal(id="fpbar"):
                yield Label("Upload as: ")
                yield Select(
                    self._KIND_LABELS,
                    value=self.default_kind,
                    allow_blank=False,
                    id="fpkind",
                )
        yield DirectoryTree(str(Path.home()), id="fptree")
        yield Footer()

    @on(DirectoryTree.FileSelected)
    async def _selected(self, event: DirectoryTree.FileSelected) -> None:
        if self.kinds is not None:
            kind = self.kinds[0]
        else:
            kind = str(self.query_one("#fpkind", Select).value)
        path = str(event.path)
        self.notify(f"Uploading {Path(path).name}…", timeout=3)
        try:
            res = await self.api.upload_file(kind, path)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        except OSError as exc:
            self.notify(f"Cannot read file: {exc}", severity="error", timeout=8)
            return
        self.dismiss(res.get("filename"))

    def action_cancel(self) -> None:
        self.dismiss(None)


# Landing & new exam ---------------------------------------------------------


class ExamListScreen(TentanatorScreen):
    """Landing screen: list exams (active or archived), create, import, archive."""

    BINDINGS = [
        ("n", "new_exam", "New exam"),
        ("u", "upload", "Upload file"),
        ("c", "combine_moodle", "Combine Moodle"),
        ("b", "global_bank", "Global bank"),
        ("w", "import_workspace", "Import workspace"),
        ("i", "import_legacy", "Import sessions"),
        ("h", "toggle_archived", "Show archived"),
        ("v", "toggle_archive_one", "Archive/restore"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.show_archived = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("Exams", id="title")
        yield ListView(id="exams")
        with Horizontal(id="actions"):
            yield Button("New [n]", id="new", variant="primary")
            yield Button("Upload [u]", id="upload")
            yield Button("Combine Moodle [c]", id="combine")
            yield Button("Global bank [b]", id="globalbank")
            yield Button("Import workspace [w]", id="impworkspace")
            yield Button("Show archived [h]", id="togglearch")
            yield Button("Refresh [r]", id="refresh")
        yield Footer()

    async def on_mount(self) -> None:
        await self.refresh_exams()

    async def refresh_exams(self) -> None:
        self.query_one("#title", Label).update("Archived exams" if self.show_archived else "Exams")
        lv = self.query_one("#exams", ListView)
        await lv.clear()
        try:
            exams = await self.api.list_exams(archived=self.show_archived)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        if not exams:
            hint = "(no archived exams)" if self.show_archived else "(no exams - press 'n')"
            await lv.append(ListItem(Label(hint)))
            return
        for e in exams:
            course = f"  [{e['course']}]" if e.get("course") else ""
            label = (
                f"{e['name']}{course}  -  {e['exam_file']}  "
                f"({e['num_questions']} q, {e['graded_count']} graded, "
                f"updated {e['last_updated'][:19]})"
            )
            await lv.append(ListItem(Label(label), name=e["name"]))

    def action_new_exam(self) -> None:
        self.app.push_screen(NewExamScreen(), self._after_new)

    def _after_new(self, name: Optional[str]) -> None:
        self.run_worker(self.refresh_exams())
        if name:
            self.app.push_screen(
                GradingScreen(name), lambda _r: self.run_worker(self.refresh_exams())
            )

    def action_upload(self) -> None:
        self.app.push_screen(FilePickerScreen())

    def action_combine_moodle(self) -> None:
        self.app.push_screen(
            CombineMoodleScreen(), lambda _r: self.run_worker(self.refresh_exams())
        )

    def action_global_bank(self) -> None:
        self.app.push_screen(GlobalBankScreen())

    def action_import_workspace(self) -> None:
        self.app.push_screen(
            WorkspaceImportScreen(), lambda _r: self.run_worker(self.refresh_exams())
        )

    async def action_import_legacy(self) -> None:
        """Import loose legacy `.tentanator_sessions/` exams into the new format."""
        try:
            result = await self.api.import_legacy_sessions()
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        imported = result.get("imported_exams", [])
        self.notify(f"Imported {len(imported)} legacy exam(s)")
        await self.refresh_exams()

    async def action_toggle_archived(self) -> None:
        self.show_archived = not self.show_archived
        label = "Show active [h]" if self.show_archived else "Show archived [h]"
        self.query_one("#togglearch", Button).label = label
        await self.refresh_exams()

    async def action_toggle_archive_one(self) -> None:
        """Archive (active view) or unarchive (archived view) the highlighted exam."""
        item = self.query_one("#exams", ListView).highlighted_child
        if item is None or not item.name:
            self.notify("Highlight an exam first", severity="warning")
            return
        name = item.name
        try:
            if self.show_archived:
                await self.api.unarchive_exam(name)
                self.notify(f"Unarchived {name}")
            else:
                await self.api.archive_exam(name)
                self.notify(f"Archived {name}")
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        await self.refresh_exams()

    async def action_refresh(self) -> None:
        await self.refresh_exams()

    @on(Button.Pressed, "#new")
    def _new(self) -> None:
        self.action_new_exam()

    @on(Button.Pressed, "#upload")
    def _upload(self) -> None:
        self.action_upload()

    @on(Button.Pressed, "#combine")
    def _combine(self) -> None:
        self.action_combine_moodle()

    @on(Button.Pressed, "#globalbank")
    def _globalbank(self) -> None:
        self.action_global_bank()

    @on(Button.Pressed, "#impworkspace")
    def _impworkspace(self) -> None:
        self.action_import_workspace()

    @on(Button.Pressed, "#togglearch")
    async def _togglearch(self) -> None:
        await self.action_toggle_archived()

    @on(Button.Pressed, "#refresh")
    async def _refresh(self) -> None:
        await self.refresh_exams()

    @on(ListView.Selected)
    def _open(self, event: ListView.Selected) -> None:
        if event.item is not None and event.item.name:
            self.app.push_screen(
                GradingScreen(event.item.name), lambda _r: self.run_worker(self.refresh_exams())
            )


class WorkspaceImportScreen(TentanatorScreen):
    """Import leftover legacy `workspaces/<name>/` folders as exams."""

    BINDINGS = [("escape", "back", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Importable legacy workspaces (Enter to import)", id="title")
        yield ListView(id="workspaces")
        yield Footer()

    async def on_mount(self) -> None:
        lv = self.query_one("#workspaces", ListView)
        await lv.clear()
        try:
            workspaces = await self.api.list_legacy_workspaces()
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        if not workspaces:
            await lv.append(ListItem(Label("(no importable workspaces)")))
            return
        for w in workspaces:
            await lv.append(
                ListItem(Label(f"{w['name']}  ({w.get('exams', 0)} exam(s))"), name=w["name"])
            )

    @on(ListView.Selected)
    async def _import(self, event: ListView.Selected) -> None:
        if event.item is None or not event.item.name:
            return
        name = event.item.name
        try:
            result = await self.api.import_workspace(name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        imported = result.get("imported_exams", [])
        self.notify(f"Imported {len(imported)} exam(s) from '{name}'")
        self.dismiss(None)

    def action_back(self) -> None:
        self.dismiss(None)


class CombineMoodleScreen(TentanatorScreen):
    """Combine two raw Moodle exports (grades + responses) into one exam file.

    The two files are uploaded as ``raw`` (into ``exams_in_raw/``) via the shared
    file picker; the backend pairs them and writes the compiled file into
    ``exams/`` where it appears in the new-exam picker.
    """

    BINDINGS = [("escape", "back", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self.grades_file: Optional[str] = None
        self.responses_file: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label("Combine Moodle dumps", id="title")
            yield Static(
                "[dim]Pick the two raw Moodle exports; the backend compiles them "
                "into one exam file under exams/.[/dim]"
            )
            with Horizontal(id="cm-grades"):
                yield Button("Pick grades file", id="pickgrades")
                yield Label("(none selected)", id="gradeslabel")
            with Horizontal(id="cm-responses"):
                yield Button("Pick responses file", id="pickresponses")
                yield Label("(none selected)", id="responseslabel")
            yield Label("Output name (optional):")
            yield Input(placeholder="auto-generated if blank", id="outputname")
            with Horizontal(id="cm-actions"):
                yield Button("Combine", id="combine", variant="primary")
                yield Button("Back [esc]", id="back")
            yield Static("", id="cmstatus")
        yield Footer()

    @on(Button.Pressed, "#pickgrades")
    def _pick_grades(self) -> None:
        self.app.push_screen(FilePickerScreen(kinds=["raw"]), self._grades_picked)

    def _grades_picked(self, filename: Optional[str]) -> None:
        if filename:
            self.grades_file = filename
            self.query_one("#gradeslabel", Label).update(filename)

    @on(Button.Pressed, "#pickresponses")
    def _pick_responses(self) -> None:
        self.app.push_screen(FilePickerScreen(kinds=["raw"]), self._responses_picked)

    def _responses_picked(self, filename: Optional[str]) -> None:
        if filename:
            self.responses_file = filename
            self.query_one("#responseslabel", Label).update(filename)

    @on(Button.Pressed, "#combine")
    async def _combine(self) -> None:
        if not self.grades_file or not self.responses_file:
            self.notify("Pick both a grades and a responses file", severity="warning")
            return
        output_name = self.query_one("#outputname", Input).value.strip() or None
        self.query_one("#cmstatus", Static).update("[dim]Combining...[/dim]")
        try:
            result = await self.api.combine_moodle(
                self.grades_file, self.responses_file, output_name
            )
        except APIError as exc:
            self.query_one("#cmstatus", Static).update("")
            self.notify(str(exc), severity="error", timeout=8)
            return
        dropped = result.get("dropped_columns", [])
        dropped_str = ", ".join(dropped) if dropped else "none"
        self.query_one("#cmstatus", Static).update(
            f"[b]Combined into {result.get('filename', '?')}[/b]  "
            f"{result.get('students', 0)} students, {result.get('questions', 0)} questions.\n"
            f"[dim]Dropped columns: {dropped_str}[/dim]\n"
            "[dim]Now available in the new-exam picker.[/dim]"
        )
        self.notify(f"Combined into {result.get('filename', '?')}", timeout=8)

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()

    def action_back(self) -> None:
        self.dismiss(None)


class GlobalBankScreen(TentanatorScreen):
    """App-wide reference question bank: status, import, reindex, semantic search."""

    BINDINGS = [("escape", "back", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label("Global question bank", id="title")
            yield Static("", id="bankstatus")
            with Horizontal(id="gb-actions"):
                yield Button("Import CSV", id="import")
                yield Button("Reindex", id="reindex")
            yield Label("Search the bank:")
            with Horizontal(id="gb-searchbar"):
                yield Input(placeholder="search query", id="searchquery")
                yield Select(
                    [("auto", "auto"), ("Swedish (se)", "se"), ("English (en)", "en")],
                    value="auto",
                    allow_blank=False,
                    id="searchlang",
                )
                yield Button("Search", id="search", variant="primary")
            yield Static("", id="searchstatus")
            yield DataTable(id="banktable")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#banktable", DataTable).add_columns(
            "Score", "QID", "Bank", "Question", "Answer"
        )
        await self._load_status()

    async def _load_status(self) -> None:
        try:
            status = await self.api.global_bank_status()
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        banks = status.get("banks", [])
        bank_str = ", ".join(f"{b['name']} ({b['questions']})" for b in banks) or "none"
        self.query_one("#bankstatus", Static).update(
            f"[b]Banks:[/b] {bank_str}\n"
            f"{status.get('total_questions', 0)} questions, "
            f"{status.get('indexed_vectors', 0)} indexed vectors"
        )

    @on(Button.Pressed, "#reindex")
    def _reindex_btn(self) -> None:
        self._reindex()

    @work
    async def _reindex(self) -> None:
        self.query_one("#searchstatus", Static).update("[dim]Reindexing...[/dim]")
        try:
            result = await self.api.global_bank_reindex()
        except APIError as exc:
            self.query_one("#searchstatus", Static).update("")
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.query_one("#searchstatus", Static).update("")
        self.notify(
            f"Embedded {result.get('embedded', 0)} of "
            f"{result.get('total_questions', 0)} questions"
        )
        await self._load_status()

    @on(Button.Pressed, "#import")
    def _import_btn(self) -> None:
        self.app.push_screen(FilePickerScreen(kinds=["raw"]), self._bank_file_picked)

    def _bank_file_picked(self, filename: Optional[str]) -> None:
        if filename:
            self._import_bank(filename)

    @work
    async def _import_bank(self, filename: str) -> None:
        self.query_one("#searchstatus", Static).update("[dim]Importing...[/dim]")
        try:
            result = await self.api.global_bank_import(filename)
        except APIError as exc:
            self.query_one("#searchstatus", Static).update("")
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.query_one("#searchstatus", Static).update("")
        self.notify(
            f"Imported {result.get('imported', 0)} questions into bank "
            f"'{result.get('bank', '?')}'. Run Reindex to embed."
        )
        await self._load_status()

    @on(Button.Pressed, "#search")
    @on(Input.Submitted, "#searchquery")
    def _search_btn(self) -> None:
        self._search()

    @work
    async def _search(self) -> None:
        query = self.query_one("#searchquery", Input).value.strip()
        if not query:
            self.notify("Enter a search query", severity="warning")
            return
        lang_val = str(self.query_one("#searchlang", Select).value)
        language = None if lang_val == "auto" else lang_val
        self.query_one("#searchstatus", Static).update("[dim]Searching...[/dim]")
        try:
            result = await self.api.global_bank_search(query, language)
        except APIError as exc:
            self.query_one("#searchstatus", Static).update("")
            self.notify(str(exc), severity="error", timeout=8)
            return
        detected = result.get("language")
        matches = result.get("matches", [])
        table = self.query_one("#banktable", DataTable)
        table.clear()
        for m in matches:
            table.add_row(
                f"{m.get('score', 0.0):.3f}",
                m.get("qid", ""),
                m.get("bank", ""),
                match_question(m, detected),
                match_answer(m, detected),
            )
        self.query_one("#searchstatus", Static).update(
            f"[dim]{len(matches)} match(es)  |  language: {detected or '?'}[/dim]"
        )

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()

    def action_back(self) -> None:
        self.dismiss(None)


class NewExamScreen(TentanatorScreen):
    """Wizard: choose an exam file then the id / input / output columns.

    Columns are pre-selected from the backend's auto-detection (`Response N` /
    `Points N` pairing); the user can adjust before creating.
    """

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
            yield Static("", id="detectinfo")
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
            with Horizontal(id="exam-actions"):
                yield Button("Create", id="create", variant="primary")
                yield Button("Re-detect", id="redetect")
                yield Button("Cancel [esc]", id="cancel")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            files = await self.api.list_exam_files()
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.query_one("#examfile", Select).set_options([(e, e) for e in files])

    @on(Select.Changed, "#examfile")
    async def _exam_chosen(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        await self._load_columns(str(event.value))

    async def _load_columns(self, file: str) -> None:
        try:
            cols = await self.api.exam_columns(file)
            detected = await self.api.detect_columns(file)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self._columns = cols
        id_set = set(detected.get("id_columns", []))
        input_set = set(detected.get("input_columns", []))
        output_set = set(detected.get("output_columns", []))
        for sel_id, chosen in (
            ("#idcols", id_set),
            ("#inputcols", input_set),
            ("#outputcols", output_set),
        ):
            sl = self.query_one(sel_id, SelectionList)
            sl.clear_options()
            sl.add_options([Selection(c, c, c in chosen) for c in cols])
        self.query_one("#detectinfo", Static).update(
            f"[dim]Auto-detected {len(output_set)} question pair(s); adjust below if needed.[/dim]"
        )

    @on(Button.Pressed, "#redetect")
    async def _redetect(self) -> None:
        value = self.query_one("#examfile", Select).value
        if value is not Select.BLANK:
            await self._load_columns(str(value))

    def action_cancel(self) -> None:
        self.dismiss(None)

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
            exam = await self.api.create_exam(payload)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.dismiss(exam["name"])


# Grading --------------------------------------------------------------------


class GradingScreen(TentanatorScreen):
    """Grade one output column at a time; hub for the per-exam tabs."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("a", "suggest", "AI suggest"),
        ("s", "skip", "Skip"),
        ("e", "export", "Export"),
        ("r", "results", "Results"),
        ("c", "scheme", "Scheme"),
        ("m", "import_grades", "Import"),
        ("p", "pdf", "PDF"),
        ("t", "qsettings", "Question settings"),
        ("g", "graded", "Graded items"),
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
        # When on, advancing to an ungraded response auto-requests an AI
        # suggestion (mirrors the web client's "auto AI-suggest" toggle).
        self.auto_suggest: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="navbar"):
            yield Button("settings [t]", id="nav-settings")
            yield Button("scheme [c]", id="nav-scheme")
            yield Button("import [m]", id="nav-import")
            yield Button("results [r]", id="nav-results")
            yield Button("pdf [p]", id="nav-pdf")
            yield Button("graded [g]", id="nav-graded")
            yield Button("+questions", id="nav-addq")
        with Horizontal(id="metabar"):
            yield Label("Course:")
            yield Input(placeholder="e.g. CS101", id="course")
        with Horizontal(id="topbar"):
            yield Select([], id="colselect", prompt="Question")
            yield Select([], id="sessionselect", prompt="Session")
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
            yield Label("auto", id="autolabel")
            yield Switch(value=self.auto_suggest, id="autosuggest")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            self.exam = await self.api.get_exam(self.exam_name)
            self.rows = await self.api.exam_rows(self.exam["exam_file"])
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.id_columns = self.exam.get("id_columns", [])
        self.query_one("#course", Input).value = self.exam.get("course") or ""
        await self.refresh_sessions()
        self._set_columns()
        if self.current_col:
            await self.refresh_question()

    def _set_columns(self) -> None:
        out_cols = self.exam.get("output_columns", [])
        sel = self.query_one("#colselect", Select)
        sel.set_options([(c, c) for c in out_cols])
        if out_cols:
            keep = self.current_col if self.current_col in out_cols else out_cols[0]
            sel.value = keep
            self.current_col = keep

    async def reload(self) -> None:
        """Re-fetch the exam after a sub-screen may have changed it."""
        try:
            self.exam = await self.api.get_exam(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.id_columns = self.exam.get("id_columns", [])
        self.query_one("#course", Input).value = self.exam.get("course") or ""
        await self.refresh_sessions()
        self._set_columns()
        if self.current_col:
            await self.refresh_question()

    def _open(self, screen: Screen) -> None:
        """Push a per-exam sub-screen and refresh on return."""
        self.app.push_screen(screen, lambda _r: self.run_worker(self.reload()))

    async def refresh_sessions(self) -> None:
        """Reload the session list and keep the active session valid."""
        try:
            sessions = await self.api.list_sessions(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        names = [s["name"] for s in sessions]
        sel = self.query_one("#sessionselect", Select)
        sel.set_options([(f"{s['name']} ({s['graded_count']})", s["name"]) for s in sessions])
        if self.active_session not in names:
            fallback = names[0] if names else "default"
            self.active_session = "default" if "default" in names else fallback
        if names:
            sel.value = self.active_session

    @on(Select.Changed, "#sessionselect")
    def _session_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self.active_session = str(event.value)

    @on(Switch.Changed, "#autosuggest")
    def _autosuggest_changed(self, event: Switch.Changed) -> None:
        self.auto_suggest = event.value
        # Turning it on mid-question kicks off a suggestion for the current row.
        if self.auto_suggest and self.suggestion is None and self.ungraded:
            self.run_worker(self.action_suggest())

    @on(Input.Submitted, "#course")
    async def _save_course(self) -> None:
        course = self.query_one("#course", Input).value.strip()
        if (self.exam.get("course") or "") == course:
            return
        try:
            updated = await self.api.update_exam(self.exam_name, {"course": course})
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.exam["course"] = updated.get("course")
        self.notify("Course saved")

    @on(Button.Pressed, "#newsession")
    def _new_session_btn(self) -> None:
        self.action_new_session()

    @work
    async def action_new_session(self) -> None:
        """Prompt for a session name (blank = auto-named) and create it."""
        name = await self.app.push_screen_wait(TextPromptScreen("New session name (blank = auto)"))
        if name is None:
            return
        try:
            created = await self.api.create_session(self.exam_name, name.strip() or None)
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
        icl_ready = False
        try:
            status = await self.api.question_status(self.exam_name, self.current_col)
            icl_ready = bool(status.get("icl_ready"))
            icl = "yes" if icl_ready else "no"
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
        # Auto AI-suggest on advance: only when enabled, ICL is ready, and we
        # have not already fetched a suggestion for this response.
        if self.auto_suggest and icl_ready and self.suggestion is None:
            await self.action_suggest()

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
            self.suggestion = await self.api.suggest(self.exam_name, self.current_col, rid)
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
            question = await self.api.grade(
                self.exam_name, self.current_col, rid, grade, self.active_session
            )
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.exam.setdefault("questions", {})[self.current_col] = question
        await self.refresh_sessions()
        # Drop the just-graded row from the queue and stay on the same index.
        del self.ungraded[self.index]
        if self.index >= len(self.ungraded):
            self.index = max(0, len(self.ungraded) - 1)
        self.suggestion = None
        await self.show_current()

    def action_skip(self) -> None:
        if self.ungraded:
            self.index = (self.index + 1) % len(self.ungraded)
            self.suggestion = None
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
            result = await self.api.sampling(self.exam_name, self.current_col, algorithm)
            self.exam = await self.api.get_exam(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(f"Selected {result.get('num_samples', 0)} representative samples")
        await self.refresh_question()

    @on(Button.Pressed, "#export")
    async def action_export(self) -> None:
        try:
            result = await self.api.export(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(f"Exported to {result.get('path', '?')}", timeout=8)

    @on(Button.Pressed, "#nav-addq")
    async def action_add_questions(self) -> None:
        """Detect Response/Points pairs in the file and add any missing questions."""
        try:
            detected = await self.api.detect_columns(self.exam["exam_file"])
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        det_out = detected.get("output_columns", [])
        det_in = detected.get("input_columns", [])
        existing_out = self.exam.get("output_columns", [])
        extra = [c for c in existing_out if c not in det_out]
        output_columns = list(det_out) + extra
        questions = self.exam.get("questions", {})
        input_columns = list(det_in) + [questions.get(c, {}).get("input_column", "") for c in extra]
        id_columns = self.exam.get("id_columns") or detected.get("id_columns", [])
        missing = [c for c in det_out if c not in existing_out]
        if not missing:
            self.notify("No additional questions detected")
            return
        try:
            self.exam = await self.api.update_exam_columns(
                self.exam_name,
                {
                    "id_columns": id_columns,
                    "input_columns": input_columns,
                    "output_columns": output_columns,
                },
            )
            self.rows = await self.api.exam_rows(self.exam["exam_file"])
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(f"Added {len(missing)} question(s)")
        self._set_columns()
        await self.refresh_question()

    @on(Button.Pressed, "#nav-settings")
    def action_qsettings(self) -> None:
        if self.current_col:
            self._open(QuestionSettingsScreen(self.exam_name, self.current_col))

    @on(Button.Pressed, "#nav-scheme")
    def action_scheme(self) -> None:
        self._open(SchemeScreen(self.exam_name))

    @on(Button.Pressed, "#nav-import")
    def action_import_grades(self) -> None:
        self._open(ImportScreen(self.exam_name))

    @on(Button.Pressed, "#nav-results")
    def action_results(self) -> None:
        self._open(ResultsScreen(self.exam_name))

    @on(Button.Pressed, "#nav-pdf")
    def action_pdf(self) -> None:
        self._open(PdfScreen(self.exam_name))

    @on(Button.Pressed, "#nav-graded")
    def action_graded(self) -> None:
        if self.current_col:
            self._open(GradedListScreen(self.exam_name, self.current_col))

    def action_back(self) -> None:
        self.dismiss(None)


class MatchPickerScreen(ModalScreen[Optional[Dict[str, Any]]]):
    """Modal list of GlobalBankMatch results; dismisses with the chosen match."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, matches: List[Dict[str, Any]], language: Optional[str]) -> None:
        super().__init__()
        self.matches = matches
        self.language = language

    def compose(self) -> ComposeResult:
        with Vertical(id="matchbox"):
            yield Label("Pick a bank match (Enter to apply)")
            yield ListView(id="matches")
            yield Button("Cancel [esc]", id="cancel")

    async def on_mount(self) -> None:
        lv = self.query_one("#matches", ListView)
        for i, m in enumerate(self.matches):
            question = match_question(m, self.language)
            label = f"{m.get('score', 0.0):.3f}  {m.get('qid', '')}  -  {question}"
            await lv.append(ListItem(Label(label), name=str(i)))
        lv.focus()

    @on(ListView.Selected)
    def _selected(self, event: ListView.Selected) -> None:
        if event.item is not None and event.item.name is not None:
            self.dismiss(self.matches[int(event.item.name)])

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class QuestionSettingsScreen(TentanatorScreen):
    """Edit a question's global-id link, exam text and sample answer."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("a", "auto_match", "Auto-match"),
    ]

    def __init__(self, exam_name: str, col: str) -> None:
        super().__init__()
        self.exam_name = exam_name
        self.col = col

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label(f"Question settings - {self.col}", id="title")
            yield Static(
                "[dim]Link the same question across exams to share graded examples.[/dim]"
            )
            with Horizontal(id="qs-match"):
                yield Button("Auto-match from bank [a]", id="automatch")
                yield Select(
                    [("auto", "auto"), ("Swedish (se)", "se"), ("English (en)", "en")],
                    value="auto",
                    allow_blank=False,
                    id="matchlang",
                )
            yield Label("Global question id:")
            yield Input(placeholder="e.g. pvt_q37_version_control", id="globalid")
            yield Label("Exam question text:")
            yield TextArea(id="examtext")
            yield Label("Sample answer (optional):")
            yield TextArea(id="sampleanswer")
            with Horizontal(id="qs-actions"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Back [esc]", id="back")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            exam = await self.api.get_exam(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        q = exam.get("questions", {}).get(self.col, {})
        self.query_one("#globalid", Input).value = q.get("global_question_id") or ""
        self.query_one("#examtext", TextArea).text = q.get("exam_question", "")
        self.query_one("#sampleanswer", TextArea).text = q.get("sample_answer", "")

    @on(Button.Pressed, "#automatch")
    def _automatch_btn(self) -> None:
        self.action_auto_match()

    @work
    async def action_auto_match(self) -> None:
        """Embed the question's answers, rank the bank and fill from a chosen match."""
        lang_val = str(self.query_one("#matchlang", Select).value)
        language = None if lang_val == "auto" else lang_val
        self.notify("Matching against the global bank...", timeout=3)
        try:
            result = await self.api.auto_match(self.exam_name, self.col, language)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        detected = result.get("language")
        matches = result.get("matches", [])
        if not matches:
            self.notify("No bank matches found", severity="warning")
            return
        chosen = await self.app.push_screen_wait(MatchPickerScreen(matches, detected))
        if chosen is None:
            return
        # Fill the fields; the user can still edit before saving.
        self.query_one("#globalid", Input).value = chosen.get("qid") or ""
        self.query_one("#examtext", TextArea).text = match_question(chosen, detected)
        self.query_one("#sampleanswer", TextArea).text = match_answer(chosen, detected)
        await self._persist()
        self.notify("Filled and saved from bank match")

    async def _persist(self) -> bool:
        """Save the current field values via put_question; returns success."""
        meta = {
            "global_question_id": self.query_one("#globalid", Input).value.strip(),
            "exam_question": self.query_one("#examtext", TextArea).text,
            "sample_answer": self.query_one("#sampleanswer", TextArea).text,
        }
        try:
            await self.api.put_question(self.exam_name, self.col, meta)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return False
        return True

    @on(Button.Pressed, "#save")
    async def _save(self) -> None:
        if await self._persist():
            self.notify("Question settings saved")
            self.dismiss(None)

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()

    def action_back(self) -> None:
        self.dismiss(None)


class GradedListScreen(TentanatorScreen):
    """List a question's graded items and remove (ungrade) a mistaken one."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("d", "remove", "Remove"),
    ]

    def __init__(self, exam_name: str, col: str) -> None:
        super().__init__()
        self.exam_name = exam_name
        self.col = col

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"Graded items - {self.col}  (Enter/d to remove)", id="title")
        yield ListView(id="graded")
        yield Footer()

    async def on_mount(self) -> None:
        await self._refresh()

    async def _refresh(self) -> None:
        lv = self.query_one("#graded", ListView)
        await lv.clear()
        try:
            exam = await self.api.get_exam(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        items = exam.get("questions", {}).get(self.col, {}).get("graded_items", [])
        if not items:
            await lv.append(ListItem(Label("(nothing graded yet)")))
            return
        for gi in items:
            await lv.append(
                ListItem(Label(f"{gi['row_id']}  ->  {gi['grade']}"), name=gi["row_id"])
            )

    @on(ListView.Selected)
    async def _selected(self, event: ListView.Selected) -> None:
        if event.item is not None and event.item.name:
            await self._ungrade(event.item.name)

    async def action_remove(self) -> None:
        item = self.query_one("#graded", ListView).highlighted_child
        if item is not None and item.name:
            await self._ungrade(item.name)

    async def _ungrade(self, rid: str) -> None:
        try:
            await self.api.ungrade(self.exam_name, self.col, rid)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(f"Removed grade for {rid}")
        await self._refresh()

    def action_back(self) -> None:
        self.dismiss(None)


class SchemeScreen(TentanatorScreen):
    """Question config table + readable grade-scheme DSL editor.

    The DSL grammar lives in the backend; this screen round-trips text through
    `/api/scheme/emit` and `/api/scheme/parse`.
    """

    BINDINGS = [("escape", "back", "Back")]

    def __init__(self, exam_name: str) -> None:
        super().__init__()
        self.exam_name = exam_name
        self.cfg_rows: List[Tuple[str, Dict[str, Input]]] = []
        # Loaded position per column, preserved on save (not editable here).
        self.cfg_pos: Dict[str, int] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label("Question config (var / group / type / max)", id="title")
            yield Vertical(id="cfgrows")
            with Horizontal(id="cfg-actions"):
                yield Button("Save config", id="saveconfig")
            yield Label("Grade scheme (DSL)")
            yield Static(
                "[dim]One per line: const name = value | name = expr | "
                "when cond -> grade | total_var: x | default_grade: x[/dim]"
            )
            yield TextArea(id="schemetext")
            with Horizontal(id="scheme-actions"):
                yield Button("Preview", id="preview")
                yield Button("Save scheme", id="savescheme", variant="primary")
            yield Static("", id="schemepreview")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            exam = await self.api.get_exam(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        questions = exam.get("questions", {})
        container = self.query_one("#cfgrows", Vertical)
        for col in exam.get("output_columns", []):
            q = questions.get(col, {})
            # The backend supplies the suggested var (Points N -> pN); show it as-is.
            self.cfg_pos[col] = int(q.get("position", 0) or 0)
            inputs = {
                "var": Input(value=q.get("var", ""), classes="cfgin"),
                "group": Input(value=q.get("group", ""), classes="cfgin"),
                "qtype": Input(value=q.get("qtype", ""), classes="cfgin"),
                "max": Input(value=str(q.get("max_points", 0) or 0), classes="cfgnum"),
            }
            self.cfg_rows.append((col, inputs))
            await container.mount(
                Horizontal(
                    Label(col, classes="cfgcol"),
                    inputs["var"],
                    inputs["group"],
                    inputs["qtype"],
                    inputs["max"],
                    classes="cfgrow",
                )
            )
        scheme = exam.get("scheme")
        if scheme:
            try:
                text = await self.api.scheme_emit(scheme)
            except APIError:
                text = ""
        else:
            text = ""
        self.query_one("#schemetext", TextArea).text = text

    def _gather_config(self) -> List[Dict[str, Any]]:
        updates: List[Dict[str, Any]] = []
        for col, inputs in self.cfg_rows:
            try:
                max_points = float(inputs["max"].value or 0)
            except ValueError:
                max_points = 0.0
            updates.append({
                "col": col,
                "var": inputs["var"].value.strip(),
                "group": inputs["group"].value.strip(),
                "qtype": inputs["qtype"].value.strip(),
                "max_points": max_points,
                "position": self.cfg_pos.get(col, 0),
            })
        return updates

    @on(Button.Pressed, "#saveconfig")
    async def _save_config(self) -> None:
        try:
            await self.api.put_questions_config(self.exam_name, self._gather_config())
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify("Question config saved")

    @on(Button.Pressed, "#preview")
    async def _preview(self) -> None:
        text = self.query_one("#schemetext", TextArea).text
        try:
            scheme = await self.api.scheme_parse(text)
            await self.api.put_questions_config(self.exam_name, self._gather_config())
            preview = await self.api.preview_results(self.exam_name, scheme)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        dist = preview.get("distribution", {})
        dist_str = "  ".join(f"{g}:{c}" for g, c in sorted(dist.items()))
        complete = preview.get("complete", 0)
        total = preview.get("total_students", 0)
        self.query_one("#schemepreview", Static).update(
            f"[b]Preview[/b]  {len(preview.get('results', []))} students  |  "
            f"complete {complete}/{total}  |  {dist_str}"
        )

    @on(Button.Pressed, "#savescheme")
    async def _save_scheme(self) -> None:
        text = self.query_one("#schemetext", TextArea).text
        try:
            scheme = await self.api.scheme_parse(text)
            await self.api.put_questions_config(self.exam_name, self._gather_config())
            await self.api.put_scheme(self.exam_name, scheme)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify("Scheme saved")

    def action_back(self) -> None:
        self.dismiss(None)


class ImportScreen(TentanatorScreen):
    """Map a graded sheet's columns to questions, preview/apply, resolve conflicts."""

    BINDINGS = [("escape", "back", "Back")]

    def __init__(self, exam_name: str) -> None:
        super().__init__()
        self.exam_name = exam_name
        self.output_columns: List[str] = []
        self.map_selects: List[Tuple[str, Select]] = []
        self.conflicts: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label("Import grades", id="title")
            yield Label("Import file (in exams/):")
            yield Select([], id="impfile", prompt="Select a file")
            yield Label("ID column in import file:")
            yield Select([], id="idcol", prompt="Select id column")
            yield Label("Map import columns to questions:")
            yield Vertical(id="mappings")
            with Horizontal(id="imp-actions"):
                yield Button("Preview", id="preview")
                yield Button("Apply", id="apply", variant="primary")
            yield Static("", id="impsummary")
            yield Label("Unresolved conflicts:")
            yield Vertical(id="conflicts")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            exam = await self.api.get_exam(self.exam_name)
            files = await self.api.list_exam_files()
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.output_columns = exam.get("output_columns", [])
        self.query_one("#impfile", Select).set_options([(f, f) for f in files])
        await self._load_conflicts()

    @on(Select.Changed, "#impfile")
    async def _file_chosen(self, event: Select.Changed) -> None:
        if event.value is Select.BLANK:
            return
        try:
            cols = await self.api.exam_columns(str(event.value))
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.query_one("#idcol", Select).set_options([(c, c) for c in cols])
        container = self.query_one("#mappings", Vertical)
        await container.remove_children()
        self.map_selects = []
        options = [("— skip —", "_skip")] + [(c, c) for c in cols]
        for out_col in self.output_columns:
            sel = Select(options, prompt="— skip —", allow_blank=False, value="_skip")
            self.map_selects.append((out_col, sel))
            await container.mount(Horizontal(Label(out_col, classes="mapcol"), sel))

    def _build_req(self) -> Optional[Dict[str, Any]]:
        file = self.query_one("#impfile", Select).value
        idcol = self.query_one("#idcol", Select).value
        if file is Select.BLANK or idcol is Select.BLANK:
            self.notify("Pick a file and id column", severity="warning")
            return None
        mappings = [
            {"column": str(sel.value), "output_col": out_col}
            for out_col, sel in self.map_selects
            if sel.value not in (Select.BLANK, "_skip")
        ]
        if not mappings:
            self.notify("Map at least one column", severity="warning")
            return None
        return {"file": str(file), "id_column": str(idcol), "mappings": mappings}

    @on(Button.Pressed, "#preview")
    async def _preview(self) -> None:
        req = self._build_req()
        if req is None:
            return
        try:
            summary = await self.api.import_preview(self.exam_name, req)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self._show_summary("Preview", summary)

    @on(Button.Pressed, "#apply")
    async def _apply(self) -> None:
        req = self._build_req()
        if req is None:
            return
        try:
            summary = await self.api.import_apply(self.exam_name, req)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self._show_summary("Applied", summary)
        await self._load_conflicts()

    def _show_summary(self, kind: str, summary: Dict[str, Any]) -> None:
        self.query_one("#impsummary", Static).update(
            f"[b]{kind}[/b]  {summary.get('new', 0)} new, {summary.get('same', 0)} unchanged, "
            f"{summary.get('conflict', 0)} conflict(s), {summary.get('skipped', 0)} skipped, "
            f"{summary.get('unknown_ids', 0)} unknown id(s)"
        )

    async def _load_conflicts(self) -> None:
        container = self.query_one("#conflicts", Vertical)
        await container.remove_children()
        try:
            self.conflicts = await self.api.get_conflicts(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        if not self.conflicts:
            await container.mount(Static("[dim]No unresolved conflicts.[/dim]"))
            return
        for i, c in enumerate(self.conflicts):
            await container.mount(
                Horizontal(
                    Label(
                        f"{c['row_id']} / {c['output_col']}: "
                        f"existing={c['existing_grade']}  incoming={c['incoming_grade']}",
                        classes="conflictlabel",
                    ),
                    Button(f"Keep {c['existing_grade']}", id=f"keep-{i}",
                           classes="conflictbtn"),
                    Button(f"Use {c['incoming_grade']}", id=f"use-{i}",
                           classes="conflictbtn"),
                    classes="conflictrow",
                )
            )

    @on(Button.Pressed, ".conflictbtn")
    async def _conflict_button(self, event: Button.Pressed) -> None:
        # Scoped to the dynamically-created conflict buttons; the static
        # #preview / #apply buttons have their own dedicated handlers.
        bid = event.button.id or ""
        if bid.startswith("keep-"):
            await self._resolve(int(bid[len("keep-"):]), "existing")
        elif bid.startswith("use-"):
            await self._resolve(int(bid[len("use-"):]), "incoming")

    async def _resolve(self, idx: int, choose: str) -> None:
        if idx >= len(self.conflicts):
            return
        c = self.conflicts[idx]
        try:
            await self.api.resolve_conflict(
                self.exam_name,
                {"output_col": c["output_col"], "row_id": c["row_id"], "choose": choose},
            )
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify("Conflict resolved")
        await self._load_conflicts()

    def action_back(self) -> None:
        self.dismiss(None)


class ResultsScreen(TentanatorScreen):
    """Computed results table, distribution and exports."""

    BINDINGS = [("escape", "back", "Back")]

    def __init__(self, exam_name: str) -> None:
        super().__init__()
        self.exam_name = exam_name

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label("Results", id="title")
            with Horizontal(id="res-actions"):
                yield Button("Export XLSX", id="exp-xlsx")
                yield Button("Export Daisy", id="exp-daisy")
                yield Button("Export CSV", id="exp-csv")
                yield Button("Results PDF", id="exp-pdf")
            yield Static("", id="summary")
            yield DataTable(id="resultstable")
        yield Footer()

    async def on_mount(self) -> None:
        table = self.query_one("#resultstable", DataTable)
        table.add_columns("ID", "Grade", "Total", "Estimate", "Complete")
        try:
            data = await self.api.get_results(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        if not data.get("has_scheme"):
            self.query_one("#summary", Static).update(
                "No grade scheme yet - set one on the Scheme tab to compute final grades."
            )
            return
        dist = data.get("distribution", {})
        dist_str = "  ".join(f"{g}:{c}" for g, c in sorted(dist.items()))
        summary = (
            f"{len(data.get('results', []))} students  |  "
            f"complete {data.get('complete', 0)}/{data.get('total_students', 0)}  |  "
            f"{data.get('unresolved_conflicts', 0)} unresolved conflict(s)  |  {dist_str}"
        )
        stats = data.get("stats")
        if stats:
            summary += (
                f"\nmean {stats.get('mean', 0):.1f}  median {stats.get('median', 0):.1f}  "
                f"min {stats.get('min', 0):.1f}  max {stats.get('max', 0):.1f}  "
                f"σ {stats.get('stdev', 0):.1f}"
            )
        self.query_one("#summary", Static).update(summary)
        for r in data.get("results", []):
            est = ", ".join(r.get("estimated", [])) or "-"
            table.add_row(
                r.get("id", ""),
                r.get("grade", "") or "-",
                f"{r.get('total', 0):.1f}",
                est,
                "yes" if r.get("complete") else "...",
            )

    async def _download(self, fn: str) -> None:
        try:
            if fn == "xlsx":
                res = await self.api.export(self.exam_name)
            elif fn == "daisy":
                res = await self.api.export_daisy(self.exam_name)
            else:
                res = await self.api.export_csv(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(f"Exported to {res.get('path', '?')}", timeout=8)

    @on(Button.Pressed, "#exp-xlsx")
    async def _xlsx(self) -> None:
        await self._download("xlsx")

    @on(Button.Pressed, "#exp-daisy")
    async def _daisy(self) -> None:
        await self._download("daisy")

    @on(Button.Pressed, "#exp-csv")
    async def _csv(self) -> None:
        await self._download("csv")

    @on(Button.Pressed, "#exp-pdf")
    def _pdf(self) -> None:
        self.app.push_screen(PdfScreen(self.exam_name))

    def action_back(self) -> None:
        self.dismiss(None)


class PdfScreen(TentanatorScreen):
    """Generate the results PDF, optionally prepending scanned cover pages."""

    BINDINGS = [("escape", "back", "Back")]

    def __init__(self, exam_name: str) -> None:
        super().__init__()
        self.exam_name = exam_name
        self.pdf_path: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label("Results PDF", id="title")
            yield Static(
                "[dim]Renders answer sheets; a matching scan prepends cover pages.[/dim]"
            )
            yield Label("Scanned exam PDF (optional):")
            yield Select([], id="scan", prompt="No scan selected")
            yield Static("", id="scaninfo")
            with Horizontal(id="pdf-actions"):
                yield Button("Generate PDF", id="gen", variant="primary")
                yield Button("With cover pages", id="gencover")
                yield Button("Upload scan", id="uploadscan")
                yield Button("Download", id="download")
            yield Static("", id="pdfstatus")
        yield Footer()

    async def on_mount(self) -> None:
        await self._load_scans()

    async def _load_scans(self) -> None:
        try:
            scans = await self.api.list_exam_scans(self.exam_name)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        eligible = [s for s in scans if s.get("matches")]
        hidden = len(scans) - len(eligible)
        needed = scans[0].get("needed") if scans else None
        self.query_one("#scan", Select).set_options(
            [(f"{s['filename']} ({s.get('covers')} pages)", s["filename"]) for s in eligible]
        )
        if eligible:
            info = f"{len(eligible)} eligible scan(s)."
            if hidden:
                info += f" {hidden} hidden (front-page count != {needed} students)."
        else:
            info = "No scan matches this exam" + (
                f" (need {needed} front pages, one per student)." if needed is not None else "."
            )
        self.query_one("#scaninfo", Static).update(f"[dim]{info}[/dim]")

    @on(Button.Pressed, "#uploadscan")
    def _upload(self) -> None:
        self.app.push_screen(
            FilePickerScreen("scans"), lambda _r: self.run_worker(self._load_scans())
        )

    @on(Button.Pressed, "#gen")
    async def _generate(self) -> None:
        await self._do_generate(None)

    @on(Button.Pressed, "#gencover")
    async def _generate_cover(self) -> None:
        scan = self.query_one("#scan", Select).value
        if scan is Select.BLANK:
            self.notify("Select an eligible scan first", severity="warning")
            return
        await self._do_generate(str(scan))

    async def _do_generate(self, scan: Optional[str]) -> None:
        self.query_one("#pdfstatus", Static).update("[dim]Generating...[/dim]")
        try:
            result = await self.api.export_results_pdf(self.exam_name, scan)
        except APIError as exc:
            self.query_one("#pdfstatus", Static).update("")
            self.notify(str(exc), severity="error", timeout=8)
            return
        path = result.get("path") if isinstance(result, dict) else None
        self.pdf_path = path if isinstance(path, str) else None
        self.query_one("#pdfstatus", Static).update(
            f"PDF generated: {self.pdf_path}" if self.pdf_path else "PDF generated."
        )

    @on(Button.Pressed, "#download")
    async def _download(self) -> None:
        if not self.pdf_path:
            self.notify("Generate a PDF first", severity="warning")
            return
        filename = self.pdf_path.split("/")[-1]
        try:
            res = await self.api.download_graded(filename)
        except APIError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        self.notify(f"Downloaded to {res.get('path', '?')}", timeout=8)

    def action_back(self) -> None:
        self.dismiss(None)


class TentanatorTUI(App):
    """Top-level app holding the shared API client."""

    CSS = """
    #title { padding: 1 2; text-style: bold; }
    #exams, #workspaces, #graded { height: 1fr; border: round $primary; margin: 0 1; }
    #actions { height: auto; padding: 1 2; }
    #navbar { height: auto; padding: 0 1; }
    #metabar { height: auto; padding: 0 1; }
    #metabar Label { padding: 1 1; }
    #topbar { height: auto; padding: 1 1; }
    #colselect { width: 28; }
    #sessionselect { width: 22; }
    #progress { padding: 1 2; }
    #responsebox { height: 1fr; border: round $primary; margin: 0 1; padding: 1; }
    #aibox { padding: 1 2; height: auto; }
    #gradebar { height: auto; padding: 1 1; }
    #gradeinput { width: 1fr; }
    #autolabel { padding: 1 0; }
    Button { margin: 0 1; }
    #promptbox {
        width: 60; height: auto; padding: 1 2;
        border: thick $primary; background: $surface;
    }
    #promptactions { height: auto; padding-top: 1; }
    .cfgrow { height: auto; padding: 0 1; }
    .cfgcol { width: 22; padding: 1 0; }
    .cfgin { width: 16; }
    .cfgnum { width: 8; }
    .mapcol { width: 30; padding: 1 0; }
    .conflictrow { height: auto; padding: 0 1; }
    .conflictlabel { width: 1fr; padding: 1 0; }
    #resultstable { height: 1fr; margin: 0 1; }
    #banktable { height: 1fr; margin: 0 1; }
    #searchquery { width: 1fr; }
    #searchlang, #matchlang { width: 22; }
    #gradeslabel, #responseslabel { padding: 1 1; }
    #matchbox {
        width: 90%; height: 80%; padding: 1 2;
        border: thick $primary; background: $surface;
    }
    #matches { height: 1fr; border: round $primary; margin: 1 0; }
    TextArea { height: 10; margin: 0 1; }
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
    """Entry point."""
    TentanatorTUI().run()


if __name__ == "__main__":
    main()
