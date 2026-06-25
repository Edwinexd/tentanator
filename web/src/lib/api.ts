// Typed client for the Tentanator Rust backend. The web app holds no grading
// logic; every call hits the API described in ../../ARCHITECTURE.md.

const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://127.0.0.1:8787'

export interface SessionSummary {
  session_name: string
  csv_file: string
  course: string | null
  last_updated: string
  num_questions: number
  archived: boolean
}

export interface WorkspaceInfo {
  name: string
  sessions: number
}

export interface ImportResult {
  imported_sessions: string[]
  imported_exams: number
  skipped_exams: number
}

export interface GradedItem {
  row_id: string
  input_text: string
  grade: string
  timestamp: string
}

export interface SamplingResult {
  algorithm: string
  selected_ids: string[]
  quality_score: number
  num_samples: number
  timestamp: string
}

export interface Question {
  question_name: string
  input_column: string
  exam_question: string
  sample_answer: string
  global_question_id: string | null
  graded_items: GradedItem[]
  sampling_result: SamplingResult | null
  var: string
  group: string
  qtype: string
  max_points: number
  position: number
  estimate?: string | null
}

export interface SchemeConst {
  name: string
  value: number
}
export interface SchemeVar {
  name: string
  expr: string
}
export interface GradeRule {
  when: string
  grade: string
}
export interface GradeScheme {
  constants: SchemeConst[]
  vars: SchemeVar[]
  rules: GradeRule[]
  total_var: string
  default_grade: string
}

export interface StudentResult {
  id: string
  grade: string
  total: number
  vars: Record<string, number>
  estimated: string[]
  complete: boolean
}
export interface ResultsResponse {
  results: StudentResult[]
  distribution: Record<string, number>
  total_students: number
  complete: number
  has_scheme: boolean
  unresolved_conflicts: number
}

export interface ColMapping {
  column: string
  output_col: string
}
export interface ImportReq {
  file: string
  id_column: string
  mappings: ColMapping[]
  label?: string
}
export interface ImportSummary {
  new: number
  same: number
  conflict: number
  skipped: number
  unknown_ids: number
  conflicts: { output_col: string; row_id: string; existing: string; incoming: string }[]
}
export interface GradeConflict {
  output_col: string
  row_id: string
  existing_grade: string
  existing_source: string
  incoming_grade: string
  incoming_source: string
  input_text: string
  timestamp: string
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

export interface Session {
  session_name: string
  csv_file: string
  id_columns: string[]
  input_columns: string[]
  output_columns: string[]
  course: string | null
  last_updated: string
  questions: Record<string, Question>
  scheme?: GradeScheme | null
}

export interface AIGradeSuggestion {
  grade: string
  reasoning_summary?: string | null
}

export interface QuestionStatus {
  graded: number
  valid_graded: number
  external: number
  icl_ready: boolean
  min_icl_examples: number
  sampling_result: SamplingResult | null
}

export type ExamRow = Record<string, string>
export type Algorithm = 'random' | 'maximin'

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

export const api = {
  listExams: () => req<string[]>('GET', '/api/exams'),
  examColumns: (file: string) => req<string[]>('GET', `/api/exams/${encodeURIComponent(file)}/columns`),
  examRows: (file: string) =>
    req<{ rows: ExamRow[] }>('GET', `/api/exams/${encodeURIComponent(file)}/rows`).then((r) => r.rows),

  listSessions: (opts: { archived?: boolean; course?: string } = {}) => {
    const params = new URLSearchParams()
    if (opts.archived) params.set('archived', 'true')
    if (opts.course) params.set('course', opts.course)
    return req<SessionSummary[]>('GET', `/api/sessions?${params.toString()}`)
  },
  createSession: (payload: {
    csv_file: string
    id_columns: string[]
    input_columns: string[]
    output_columns: string[]
    name?: string
    course?: string
  }) => req<Session>('POST', '/api/sessions', payload),
  getSession: (name: string) => req<Session>('GET', `/api/sessions/${encodeURIComponent(name)}`),
  updateSession: (name: string, meta: { course?: string }) =>
    req<Session>('PUT', `/api/sessions/${encodeURIComponent(name)}`, meta),

  listLegacyWorkspaces: () => req<WorkspaceInfo[]>('GET', '/api/legacy-workspaces'),
  importWorkspace: (name: string) =>
    req<ImportResult>('POST', `/api/legacy-workspaces/${encodeURIComponent(name)}/import`),

  putQuestion: (name: string, col: string, meta: Record<string, unknown>) =>
    req<Question>('PUT', `/api/sessions/${encodeURIComponent(name)}/questions/${encodeURIComponent(col)}`, meta),
  sampling: (name: string, col: string, algorithm: Algorithm, nSamples?: number) =>
    req<SamplingResult>(
      'POST',
      `/api/sessions/${encodeURIComponent(name)}/questions/${encodeURIComponent(col)}/sampling`,
      nSamples !== undefined ? { algorithm, n_samples: nSamples } : { algorithm },
    ),
  grade: (name: string, col: string, rowId: string, grade: string) =>
    req<Question>(
      'POST',
      `/api/sessions/${encodeURIComponent(name)}/questions/${encodeURIComponent(col)}/grade`,
      { row_id: rowId, grade },
    ),
  suggest: (name: string, col: string, rowId: string) =>
    req<AIGradeSuggestion>(
      'POST',
      `/api/sessions/${encodeURIComponent(name)}/questions/${encodeURIComponent(col)}/suggest`,
      { row_id: rowId },
    ),
  questionStatus: (name: string, col: string) =>
    req<QuestionStatus>(
      'GET',
      `/api/sessions/${encodeURIComponent(name)}/questions/${encodeURIComponent(col)}/status`,
    ),
  exportSession: (name: string) =>
    req<{ path: string }>('POST', `/api/sessions/${encodeURIComponent(name)}/export`),
  exportDaisy: (name: string) =>
    req<{ path: string }>('POST', `/api/sessions/${encodeURIComponent(name)}/export/daisy`),
  exportCsv: (name: string) =>
    req<{ path: string }>('POST', `/api/sessions/${encodeURIComponent(name)}/export/csv`),
  listScans: () => req<string[]>('GET', '/api/scans'),
  exportResultsPdf: (name: string, scanned_pdf?: string) =>
    req<{ path: string; students: number; covers_missing: string[] }>(
      'POST',
      `/api/sessions/${encodeURIComponent(name)}/export/results-pdf`,
      { scanned_pdf: scanned_pdf || null },
    ),

  putQuestionsConfig: (name: string, updates: QuestionConfigUpdate[]) =>
    req<Session>('PUT', `/api/sessions/${encodeURIComponent(name)}/questions-config`, updates),
  putScheme: (name: string, scheme: GradeScheme) =>
    req<void>('PUT', `/api/sessions/${encodeURIComponent(name)}/scheme`, scheme),
  getResults: (name: string) =>
    req<ResultsResponse>('GET', `/api/sessions/${encodeURIComponent(name)}/results`),
  previewResults: (name: string, scheme: GradeScheme) =>
    req<ResultsResponse>('POST', `/api/sessions/${encodeURIComponent(name)}/results`, scheme),

  importPreview: (name: string, body: ImportReq) =>
    req<ImportSummary>('POST', `/api/sessions/${encodeURIComponent(name)}/import/preview`, body),
  importApply: (name: string, body: ImportReq) =>
    req<ImportSummary>('POST', `/api/sessions/${encodeURIComponent(name)}/import/apply`, body),
  getConflicts: (name: string) =>
    req<GradeConflict[]>('GET', `/api/sessions/${encodeURIComponent(name)}/conflicts`),
  resolveConflict: (name: string, body: { output_col: string; row_id: string; choose: string }) =>
    req<void>('POST', `/api/sessions/${encodeURIComponent(name)}/conflicts/resolve`, body),
}

export function rowId(row: ExamRow, idColumns: string[]): string {
  return idColumns.map((c) => row[c] ?? '').join('_')
}

export function isMeaningful(text: string): boolean {
  const t = (text ?? '').trim()
  return t !== '' && t !== '-' && t !== 'N/A'
}
