// Typed client for the Tentanator Rust backend. The web app holds no grading
// logic; every call hits the API described in ../../ARCHITECTURE.md.
//
// An exam is the central object; sessions are lightweight grading passes under
// an exam. Exam *files* on disk live under /api/exam-files.
//
// The domain DTOs below are GENERATED from the Rust backend structs via ts-rs
// (see web/src/lib/generated/). Do not hand-edit them — change the Rust struct
// and regenerate (`cd backend && cargo test export_bindings`). Request/response
// transport shapes that have no dedicated Rust struct stay hand-written here.

import type { AIGradeSuggestion } from './generated/AIGradeSuggestion'
import type { Exam } from './generated/Exam'
import type { ExamSummary } from './generated/ExamSummary'
import type { GradeConflict } from './generated/GradeConflict'
import type { GradeRule } from './generated/GradeRule'
import type { GradeScheme } from './generated/GradeScheme'
import type { GradeStats } from './generated/GradeStats'
import type { GlobalBankMatch } from './generated/GlobalBankMatch'
import type { GradedItem } from './generated/GradedItem'
import type { ImportResult } from './generated/ImportResult'
import type { QuestionGrades } from './generated/QuestionGrades'
import type { SamplingResult } from './generated/SamplingResult'
import type { SchemeConst } from './generated/SchemeConst'
import type { SchemeVar } from './generated/SchemeVar'
import type { Session } from './generated/Session'
import type { SessionSummary } from './generated/SessionSummary'
import type { StudentResult } from './generated/StudentResult'
import type { WorkspaceInfo } from './generated/WorkspaceInfo'

export type {
  AIGradeSuggestion,
  Exam,
  ExamSummary,
  GradeConflict,
  GradeRule,
  GradeScheme,
  GradeStats,
  GlobalBankMatch,
  GradedItem,
  ImportResult,
  QuestionGrades,
  SamplingResult,
  SchemeConst,
  SchemeVar,
  Session,
  SessionSummary,
  StudentResult,
  WorkspaceInfo,
}

const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://127.0.0.1:8787'

// ---------------------------------------------------------------------------
// Response structs (transport shapes without a dedicated Rust struct)
// ---------------------------------------------------------------------------

export interface ResultsResponse {
  results: StudentResult[]
  distribution: Record<string, number>
  stats: GradeStats | null
  total_students: number
  complete: number
  has_scheme: boolean
  unresolved_conflicts: number
}

export interface ColMapping {
  column: string
  output_col: string
}
export interface ConflictSample {
  output_col: string
  row_id: string
  existing: string
  incoming: string
}
export interface ImportSummary {
  new: number
  same: number
  conflict: number
  skipped: number
  unknown_ids: number
  conflicts: ConflictSample[]
}
export interface QuestionConfigUpdate {
  col: string
  var: string
  group: string
  qtype: string
  max_points: number
  position: number
  estimate?: string
}

export interface QuestionStatus {
  graded: number
  valid_graded: number
  external: number
  icl_ready: boolean
  min_icl_examples: number
  sampling_result: SamplingResult | null
}

export interface RenderQuestion {
  label: string
  group: string
  qtype: string
  response: string
  points: number | null
  max: number
  estimated: boolean
}
export interface RenderStudent {
  id: string
  grade: string
  total: number
  questions: RenderQuestion[]
}
export interface RenderData {
  exam: string
  students: RenderStudent[]
}

export interface ScanMatch {
  filename: string
  covers: number | null
  needed: number
  matches: boolean
}

export interface DetectedColumns {
  id_columns: string[]
  input_columns: string[]
  output_columns: string[]
}

export interface SchemeText {
  text: string
}

export interface CombineMoodleResp {
  filename: string
  students: number
  questions: number
  dropped_columns: string[]
}

export interface GlobalBankInfo {
  name: string
  questions: number
}
export interface GlobalBankStatus {
  banks: GlobalBankInfo[]
  total_questions: number
  indexed_vectors: number
}
export interface GlobalBankReindexResp {
  embedded: number
  total_questions: number
}
export interface GlobalBankMatches {
  language: string
  matches: GlobalBankMatch[]
}

// ---------------------------------------------------------------------------
// Request structs
// ---------------------------------------------------------------------------

export interface CreateExam {
  exam_file: string
  id_columns: string[]
  input_columns: string[]
  output_columns: string[]
  name?: string
  course?: string
}

export interface ExamMeta {
  course: string | null
}

export interface ColumnsReq {
  id_columns: string[]
  input_columns: string[]
  output_columns: string[]
}

export interface CreateSessionReq {
  name?: string
}

export interface QuestionMeta {
  exam_question?: string | null
  sample_answer?: string | null
  global_question_id?: string | null
}

export interface SamplingReq {
  algorithm: Algorithm
  n_samples?: number
}

export interface GradeReq {
  row_id: string
  grade: string
  session?: string
}

export interface SuggestReq {
  row_id: string
}

export interface ImportReq {
  file: string
  id_column: string
  mappings: ColMapping[]
  label?: string
}

export interface ResolveReq {
  output_col: string
  row_id: string
  choose: string
}

export interface ResultsPdfReq {
  scanned_pdf?: string | null
}

export type ExamRow = Record<string, string>
export type Algorithm = 'random' | 'maximin'

// ---------------------------------------------------------------------------
// Plumbing
// ---------------------------------------------------------------------------

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body !== undefined ? { 'content-type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let message = res.statusText
    try {
      const data = (await res.json()) as { error?: string }
      if (data.error) message = data.error
    } catch {
      // non-JSON error body
    }
    throw new Error(`HTTP ${res.status}: ${message}`)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

const enc = encodeURIComponent

async function triggerDownload(method: string, path: string): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, { method })
  if (!res.ok) {
    let message = res.statusText
    try {
      const d = (await res.json()) as { error?: string }
      if (d.error) message = d.error
    } catch {
      /* */
    }
    throw new Error(`HTTP ${res.status}: ${message}`)
  }
  const blob = await res.blob()
  const cd = res.headers.get('content-disposition') ?? ''
  const m = /filename="?([^"]+)"?/.exec(cd)
  const filename = m ? m[1] : 'download'
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

async function uploadBinary(
  kind: 'exams' | 'scans' | 'raw',
  file: File,
): Promise<{ filename: string }> {
  const res = await fetch(`${API_BASE}/api/files/${kind}/${enc(file.name)}`, {
    method: 'PUT',
    body: file,
  })
  if (!res.ok) {
    let message = res.statusText
    try {
      const d = (await res.json()) as { error?: string }
      if (d.error) message = d.error
    } catch {
      // non-JSON
    }
    throw new Error(`HTTP ${res.status}: ${message}`)
  }
  return (await res.json()) as { filename: string }
}

export const api = {
  // --- health ---
  health: () => req<{ status: string }>('GET', '/api/health'),

  // --- exam files on disk ---
  listExamFiles: () => req<string[]>('GET', '/api/exam-files'),
  listScans: () => req<string[]>('GET', '/api/scans'),
  listExamScans: (name: string) => req<ScanMatch[]>('GET', `/api/exams/${enc(name)}/scans`),
  examColumns: (file: string) => req<string[]>('GET', `/api/exam-files/${enc(file)}/columns`),
  examRows: (file: string) =>
    req<{ rows: ExamRow[] }>('GET', `/api/exam-files/${enc(file)}/rows`).then((r) => r.rows),
  detectColumns: (file: string) =>
    req<DetectedColumns>('GET', `/api/exam-files/${enc(file)}/detect`),
  uploadExamFile: (file: File) => uploadBinary('exams', file),
  uploadScan: (file: File) => uploadBinary('scans', file),
  uploadRawFile: (file: File) => uploadBinary('raw', file),
  // Combine two raw Moodle dumps (grades + responses) into one exam file.
  combineMoodle: (gradesFile: string, responsesFile: string, outputName?: string) =>
    req<CombineMoodleResp>('POST', '/api/exam-files/combine-moodle', {
      grades_file: gradesFile,
      responses_file: responsesFile,
      output_name: outputName,
    }),

  // --- exams (the central entity) ---
  listExams: (opts: { archived?: boolean; course?: string } = {}) => {
    const params = new URLSearchParams()
    if (opts.archived) params.set('archived', 'true')
    if (opts.course) params.set('course', opts.course)
    return req<ExamSummary[]>('GET', `/api/exams?${params.toString()}`)
  },
  createExam: (payload: CreateExam) => req<Exam>('POST', '/api/exams', payload),
  getExam: (name: string) => req<Exam>('GET', `/api/exams/${enc(name)}`),
  updateExam: (name: string, meta: ExamMeta) =>
    req<Exam>('PUT', `/api/exams/${enc(name)}`, meta),
  updateExamColumns: (name: string, body: ColumnsReq) =>
    req<Exam>('PUT', `/api/exams/${enc(name)}/columns`, body),
  archiveExam: (name: string) => req<void>('POST', `/api/exams/${enc(name)}/archive`),
  unarchiveExam: (name: string) => req<void>('POST', `/api/exams/${enc(name)}/unarchive`),
  deleteExam: (name: string) => req<void>('DELETE', `/api/exams/${enc(name)}`),

  // --- legacy import (old Python-app data -> new format, on demand) ---
  listLegacyWorkspaces: () => req<WorkspaceInfo[]>('GET', '/api/legacy-workspaces'),
  importWorkspace: (name: string) =>
    req<ImportResult>('POST', `/api/legacy-workspaces/${enc(name)}/import`),
  legacySessionsInfo: () => req<{ count: number }>('GET', '/api/legacy-sessions'),
  importLegacySessions: () =>
    req<{ imported_exams: string[] }>('POST', '/api/legacy-sessions/import'),

  // --- sessions (grading passes under an exam) ---
  listSessions: (exam: string) =>
    req<SessionSummary[]>('GET', `/api/exams/${enc(exam)}/sessions`),
  createSession: (exam: string, name?: string) =>
    req<Session>('POST', `/api/exams/${enc(exam)}/sessions`, { name } satisfies CreateSessionReq),
  deleteSession: (exam: string, session: string) =>
    req<void>('DELETE', `/api/exams/${enc(exam)}/sessions/${enc(session)}`),

  // --- questions & grading ---
  putQuestion: (name: string, col: string, meta: QuestionMeta) =>
    req<QuestionGrades>('PUT', `/api/exams/${enc(name)}/questions/${enc(col)}`, meta),
  sampling: (name: string, col: string, algorithm: Algorithm, nSamples?: number) =>
    req<SamplingResult>(
      'POST',
      `/api/exams/${enc(name)}/questions/${enc(col)}/sampling`,
      (nSamples !== undefined
        ? { algorithm, n_samples: nSamples }
        : { algorithm }) satisfies SamplingReq,
    ),
  grade: (name: string, col: string, rowId: string, grade: string, session?: string) =>
    req<QuestionGrades>('POST', `/api/exams/${enc(name)}/questions/${enc(col)}/grade`, {
      row_id: rowId,
      grade,
      session,
    } satisfies GradeReq),
  ungrade: (name: string, col: string, rowId: string) =>
    req<QuestionGrades>(
      'DELETE',
      `/api/exams/${enc(name)}/questions/${enc(col)}/grade/${enc(rowId)}`,
    ),
  suggest: (name: string, col: string, rowId: string) =>
    req<AIGradeSuggestion>('POST', `/api/exams/${enc(name)}/questions/${enc(col)}/suggest`, {
      row_id: rowId,
    } satisfies SuggestReq),
  questionStatus: (name: string, col: string) =>
    req<QuestionStatus>('GET', `/api/exams/${enc(name)}/questions/${enc(col)}/status`),
  autoMatch: (name: string, col: string, language?: string, topK?: number) =>
    req<GlobalBankMatches>(
      'POST',
      `/api/exams/${enc(name)}/questions/${enc(col)}/auto-match`,
      { language: language ?? null, top_k: topK ?? null },
    ),

  // --- global question bank (app-wide; not exam/course-scoped) ---
  globalBankStatus: () => req<GlobalBankStatus>('GET', '/api/global-bank'),
  globalBankReindex: () => req<GlobalBankReindexResp>('POST', '/api/global-bank/reindex'),
  // Import a bank CSV (already uploaded via uploadRawFile) into the DB.
  globalBankImport: (file: string, bank?: string) =>
    req<{ bank: string; imported: number }>('POST', '/api/global-bank/import', {
      file,
      bank: bank ?? null,
    }),
  globalBankSearch: (query: string, language?: string, topK?: number) =>
    req<GlobalBankMatches>('POST', '/api/global-bank/search', {
      query,
      language: language ?? null,
      top_k: topK ?? null,
    }),

  // --- scheme, config & results ---
  // The readable scheme DSL grammar lives in the backend; clients round-trip
  // through these instead of parsing/emitting it themselves.
  schemeParse: (text: string) => req<GradeScheme>('POST', '/api/scheme/parse', { text }),
  schemeEmit: (scheme: GradeScheme) =>
    req<SchemeText>('POST', '/api/scheme/emit', scheme).then((r) => r.text),
  putScheme: (name: string, scheme: GradeScheme) =>
    req<void>('PUT', `/api/exams/${enc(name)}/scheme`, scheme),
  putQuestionsConfig: (name: string, updates: QuestionConfigUpdate[]) =>
    req<Exam>('PUT', `/api/exams/${enc(name)}/questions-config`, updates),
  getResults: (name: string) => req<ResultsResponse>('GET', `/api/exams/${enc(name)}/results`),
  previewResults: (name: string, scheme: GradeScheme) =>
    req<ResultsResponse>('POST', `/api/exams/${enc(name)}/results`, scheme),
  renderData: (name: string) =>
    req<RenderData>('GET', `/api/exams/${enc(name)}/render-data`),

  // --- imports & conflicts ---
  importPreview: (name: string, body: ImportReq) =>
    req<ImportSummary>('POST', `/api/exams/${enc(name)}/import/preview`, body),
  importApply: (name: string, body: ImportReq) =>
    req<ImportSummary>('POST', `/api/exams/${enc(name)}/import/apply`, body),
  getConflicts: (name: string) =>
    req<GradeConflict[]>('GET', `/api/exams/${enc(name)}/conflicts`),
  resolveConflict: (name: string, body: ResolveReq) =>
    req<void>('POST', `/api/exams/${enc(name)}/conflicts/resolve`, body),

  // --- exports & downloads ---
  exportExam: (name: string) => triggerDownload('POST', `/api/exams/${enc(name)}/export`),
  exportDaisy: (name: string) => triggerDownload('POST', `/api/exams/${enc(name)}/export/daisy`),
  exportCsv: (name: string) => triggerDownload('POST', `/api/exams/${enc(name)}/export/csv`),
  downloadGraded: (filename: string) => triggerDownload('GET', `/api/graded/${enc(filename)}`),
  // POST export/results-pdf returns opaque JSON proxied from the renderer
  // service (NOT a file download).
  exportResultsPdf: (name: string, scannedPdf?: string) =>
    req<Record<string, unknown>>('POST', `/api/exams/${enc(name)}/export/results-pdf`, {
      scanned_pdf: scannedPdf ?? null,
    } satisfies ResultsPdfReq),
}

export function rowId(row: ExamRow, idColumns: string[]): string {
  return idColumns.map((c) => row[c] ?? '').join('_')
}

export function isMeaningful(text: string): boolean {
  const t = (text ?? '').trim()
  return t !== '' && t !== '-' && t !== 'N/A'
}
