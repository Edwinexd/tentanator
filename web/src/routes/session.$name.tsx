import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useMemo, useState } from 'react'
import {
  api,
  isMeaningful,
  rowId,
  type AIGradeSuggestion,
  type Algorithm,
  type ExamRow,
  type QuestionStatus,
  type Session,
} from '#/lib/api'

export const Route = createFileRoute('/session/$name')({ component: SessionView })

function SessionView() {
  const { name } = Route.useParams()
  const [session, setSession] = useState<Session | null>(null)
  const [rows, setRows] = useState<ExamRow[]>([])
  const [col, setCol] = useState('')
  const [index, setIndex] = useState(0)
  const [gradeValue, setGradeValue] = useState('')
  const [suggestion, setSuggestion] = useState<AIGradeSuggestion | null>(null)
  const [status, setStatus] = useState<QuestionStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api
      .getSession(name)
      .then((s) => {
        setSession(s)
        setRows([])
        setCol(s.output_columns[0] ?? '')
        return api.examRows(s.csv_file)
      })
      .then(setRows)
      .catch((e: Error) => setError(e.message))
  }, [name])

  const question = session && col ? session.questions[col] : undefined
  const inputCol = question?.input_column ?? ''

  const ungraded = useMemo(() => {
    if (!question) return [] as ExamRow[]
    const idCols = session?.id_columns ?? []
    const gradedIds = new Set(question.graded_items.map((g) => g.row_id))
    const priority = new Set(question.sampling_result?.selected_ids ?? [])
    return rows
      .filter((r) => !gradedIds.has(rowId(r, idCols)) && isMeaningful(r[inputCol] ?? ''))
      .sort(
        (a, b) =>
          Number(!priority.has(rowId(a, idCols))) - Number(!priority.has(rowId(b, idCols))),
      )
  }, [question, rows, session, inputCol])

  const safeIndex = ungraded.length ? Math.min(index, ungraded.length - 1) : 0
  const current = ungraded[safeIndex]

  useEffect(() => {
    setIndex(0)
    setSuggestion(null)
    setGradeValue('')
  }, [col])

  useEffect(() => {
    if (col) api.questionStatus(name, col).then(setStatus).catch(() => setStatus(null))
  }, [name, col, session])

  async function save() {
    if (!current || !col) return
    const grade = gradeValue.trim()
    if (!grade) return setError('Enter a grade (or request an AI suggestion)')
    setError(null)
    setBusy(true)
    try {
      const updated = await api.grade(name, col, rowId(current, session?.id_columns ?? []), grade)
      setSession((s) => (s ? { ...s, questions: { ...s.questions, [col]: updated } } : s))
      setGradeValue('')
      setSuggestion(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function suggest() {
    if (!current || !col) return
    setBusy(true)
    setError(null)
    try {
      const s = await api.suggest(name, col, rowId(current, session?.id_columns ?? []))
      setSuggestion(s)
      setGradeValue(s.grade)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function sample(algorithm: Algorithm) {
    if (!col) return
    setBusy(true)
    setInfo(null)
    setError(null)
    try {
      const result = await api.sampling(name, col, algorithm)
      const fresh = await api.getSession(name)
      setSession(fresh)
      setInfo(`Selected ${result.num_samples} representative sample(s) via ${algorithm}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function exportSession() {
    setBusy(true)
    try {
      const { path } = await api.exportSession(name)
      setInfo(`Exported to ${path}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (!session) {
    return (
      <div className="mx-auto max-w-3xl p-8">
        {error ? (
          <p className="rounded bg-red-100 p-3 text-red-700">{error}</p>
        ) : (
          <p className="text-gray-500">Loading…</p>
        )}
        <Link to="/" className="mt-4 inline-block text-blue-600 hover:underline">
          ← back
        </Link>
      </div>
    )
  }

  const gradedCount = question?.graded_items.length ?? 0

  return (
    <div className="mx-auto max-w-3xl space-y-4 p-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{session.session_name}</h1>
        <Link to="/" className="text-blue-600 hover:underline">
          ← sessions
        </Link>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <select
          className="rounded border p-2"
          value={col}
          onChange={(e) => setCol(e.target.value)}
        >
          {session.output_columns.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <button
          disabled={busy}
          onClick={() => sample('random')}
          className="rounded border px-3 py-2 hover:bg-gray-50 disabled:opacity-50"
        >
          random sample
        </button>
        <button
          disabled={busy}
          onClick={() => sample('maximin')}
          className="rounded border px-3 py-2 hover:bg-gray-50 disabled:opacity-50"
        >
          maximin sample
        </button>
        <button
          disabled={busy}
          onClick={exportSession}
          className="rounded border px-3 py-2 hover:bg-gray-50 disabled:opacity-50"
        >
          export
        </button>
      </div>

      <div className="text-sm text-gray-600">
        graded {gradedCount}/{rows.length} · {ungraded.length} ungraded ·{' '}
        ICL ready: {status ? (status.icl_ready ? 'yes' : 'no') : '…'}
      </div>

      {info && <p className="rounded bg-green-100 p-2 text-green-800">{info}</p>}
      {error && <p className="rounded bg-red-100 p-2 text-red-700">{error}</p>}

      {current ? (
        <>
          <div className="text-sm text-gray-500">
            {safeIndex + 1}/{ungraded.length} ungraded · id:{' '}
            {rowId(current, session.id_columns)}
          </div>
          <div className="max-h-80 overflow-auto whitespace-pre-wrap rounded border bg-gray-50 p-4">
            {current[inputCol]}
          </div>

          {suggestion && (
            <div className="rounded border-l-4 border-blue-500 bg-blue-50 p-3">
              <div className="font-medium">AI grade: {suggestion.grade}</div>
              {suggestion.reasoning_summary && (
                <div className="mt-1 text-sm text-gray-600">{suggestion.reasoning_summary}</div>
              )}
            </div>
          )}

          <div className="flex items-center gap-2">
            <input
              className="flex-1 rounded border p-2"
              placeholder="grade e.g. 7.5 or 2+1.5"
              value={gradeValue}
              onChange={(e) => setGradeValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void save()
              }}
            />
            <button
              disabled={busy}
              onClick={suggest}
              className="rounded border px-3 py-2 hover:bg-gray-50 disabled:opacity-50"
            >
              AI suggest
            </button>
            <button
              disabled={busy}
              onClick={save}
              className="rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              Save
            </button>
            <button
              disabled={busy || ungraded.length < 2}
              onClick={() => setIndex((i) => (i + 1) % ungraded.length)}
              className="rounded border px-3 py-2 hover:bg-gray-50 disabled:opacity-50"
            >
              Skip
            </button>
          </div>
        </>
      ) : (
        <p className="text-gray-500">Nothing left to grade for this question.</p>
      )}
    </div>
  )
}
