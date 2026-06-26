// Typed client for the Tentanator Rust backend. The web app holds no grading
// logic; every call hits the API described in ../../ARCHITECTURE.md.
//
// An exam is the central object; sessions are lightweight grading passes under
// an exam. Exam *files* on disk live under /api/exam-files.

const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://127.0.0.1:8787'

export interface ExamSummary {
  name: string
  exam_file: string
  course: string | null
  last_updated: string
  num_questions: number
  archived: boolean
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

export interface Exam {
  name: string
  exam_file: string
  id_columns: string[]
  input_columns: string[]
  output_columns: string[]
  course: string | null
  last_updated: string
  questions: Record<string, Question>
  scheme?: GradeScheme | null
}

export interface Session {
  exam: string
  name: string
  created_at: string
  last_updated: string
}

export interface SessionSummary {
  exam: string
  name: string
  created_at: string
  last_updated: string
  graded_count: number
}

export interface WorkspaceInfo {
  name: string
  exams: number
}
export interface ImportResult {
  imported_exams: string[]
  imported_files: number
  skipped_files: number
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

export const api = {
  // --- exam files on disk ---
  listExamFiles: () => req<string[]>('GET', '/api/exam-files'),
  examColumns: (file: string) => req<string[]>('GET', `/api/exam-files/${enc(file)}/columns`),
  examRows: (file: string) =>
    req<{ rows: ExamRow[] }>('GET', `/api/exam-files/${enc(file)}/rows`).then((r) => r.rows),

  // --- exams (the central entity) ---
  listExams: (opts: { archived?: boolean; course?: string } = {}) => {
    const params = new URLSearchParams()
    if (opts.archived) params.set('archived', 'true')
    if (opts.course) params.set('course', opts.course)
    return req<ExamSummary[]>('GET', `/api/exams?${params.toString()}`)
  },
  createExam: (payload: {
    exam_file: string
    id_columns: string[]
    input_columns: string[]
    output_columns: string[]
    name?: string
    course?: string
  }) => req<Exam>('POST', '/api/exams', payload),
  getExam: (name: string) => req<Exam>('GET', `/api/exams/${enc(name)}`),
  updateExam: (name: string, meta: { course?: string }) =>
    req<Exam>('PUT', `/api/exams/${enc(name)}`, meta),
  updateExamColumns: (
    name: string,
    body: { id_columns: string[]; input_columns: string[]; output_columns: string[] },
  ) => req<Exam>('PUT', `/api/exams/${enc(name)}/columns`, body),
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
    req<Session>('POST', `/api/exams/${enc(exam)}/sessions`, { name }),
  deleteSession: (exam: string, session: string) =>
    req<void>('DELETE', `/api/exams/${enc(exam)}/sessions/${enc(session)}`),

  // --- questions & grading ---
  putQuestion: (name: string, col: string, meta: Record<string, unknown>) =>
    req<Question>('PUT', `/api/exams/${enc(name)}/questions/${enc(col)}`, meta),
  sampling: (name: string, col: string, algorithm: Algorithm, nSamples?: number) =>
    req<SamplingResult>(
      'POST',
      `/api/exams/${enc(name)}/questions/${enc(col)}/sampling`,
      nSamples !== undefined ? { algorithm, n_samples: nSamples } : { algorithm },
    ),
  grade: (name: string, col: string, rowId: string, grade: string, session?: string) =>
    req<Question>('POST', `/api/exams/${enc(name)}/questions/${enc(col)}/grade`, {
      row_id: rowId,
      grade,
      session,
    }),
  suggest: (name: string, col: string, rowId: string) =>
    req<AIGradeSuggestion>('POST', `/api/exams/${enc(name)}/questions/${enc(col)}/suggest`, {
      row_id: rowId,
    }),
  questionStatus: (name: string, col: string) =>
    req<QuestionStatus>('GET', `/api/exams/${enc(name)}/questions/${enc(col)}/status`),

  // --- exports & results ---
  exportExam: (name: string) => triggerDownload('POST', `/api/exams/${enc(name)}/export`),
  exportDaisy: (name: string) => triggerDownload('POST', `/api/exams/${enc(name)}/export/daisy`),
  exportCsv: (name: string) => triggerDownload('POST', `/api/exams/${enc(name)}/export/csv`),
  downloadGraded: (filename: string) => triggerDownload('GET', `/api/graded/${enc(filename)}`),
  listScans: () => req<string[]>('GET', '/api/scans'),
  uploadFile: async (kind: 'exams' | 'scans', file: File): Promise<{ filename: string }> => {
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
  },
  exportResultsPdf: (name: string, scanned_pdf?: string) =>
    req<{ path: string; students: number; covers_missing: string[] }>(
      'POST',
      `/api/exams/${enc(name)}/export/results-pdf`,
      { scanned_pdf: scanned_pdf || null },
    ),

  putQuestionsConfig: (name: string, updates: QuestionConfigUpdate[]) =>
    req<Exam>('PUT', `/api/exams/${enc(name)}/questions-config`, updates),
  putScheme: (name: string, scheme: GradeScheme) =>
    req<void>('PUT', `/api/exams/${enc(name)}/scheme`, scheme),
  getResults: (name: string) => req<ResultsResponse>('GET', `/api/exams/${enc(name)}/results`),
  previewResults: (name: string, scheme: GradeScheme) =>
    req<ResultsResponse>('POST', `/api/exams/${enc(name)}/results`, scheme),

  importPreview: (name: string, body: ImportReq) =>
    req<ImportSummary>('POST', `/api/exams/${enc(name)}/import/preview`, body),
  importApply: (name: string, body: ImportReq) =>
    req<ImportSummary>('POST', `/api/exams/${enc(name)}/import/apply`, body),
  getConflicts: (name: string) =>
    req<GradeConflict[]>('GET', `/api/exams/${enc(name)}/conflicts`),
  resolveConflict: (name: string, body: { output_col: string; row_id: string; choose: string }) =>
    req<void>('POST', `/api/exams/${enc(name)}/conflicts/resolve`, body),
}

export function rowId(row: ExamRow, idColumns: string[]): string {
  return idColumns.map((c) => row[c] ?? '').join('_')
}

export function isMeaningful(text: string): boolean {
  const t = (text ?? '').trim()
  return t !== '' && t !== '-' && t !== 'N/A'
}

export function detectQuestionPairs(columns: string[]): {
  id_columns: string[]
  input_columns: string[]
  output_columns: string[]
} {
  const inputs = new Map<number, string>()
  const outputs = new Map<number, string>()
  for (const c of columns) {
    const t = c.trim()
    let m = /^response\s*(\d+)$/i.exec(t)
    if (m) {
      inputs.set(Number(m[1]), c)
      continue
    }
    m = /^points?\s*(\d+)$/i.exec(t)
    if (m) outputs.set(Number(m[1]), c)
  }
  const ns = [...inputs.keys()].filter((n) => outputs.has(n)).sort((a, b) => a - b)
  const id =
    columns.find((c) => /daisy\s*id/i.test(c)) ??
    columns.find((c) => /^id$/i.test(c)) ??
    columns[0]
  return {
    id_columns: id ? [id] : [],
    input_columns: ns.map((n) => inputs.get(n)!),
    output_columns: ns.map((n) => outputs.get(n)!),
  }
}
