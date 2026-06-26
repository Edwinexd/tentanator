import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useMemo, useState } from 'react'
import {
  api,
  isMeaningful,
  rowId,
  type AIGradeSuggestion,
  type Algorithm,
  type Exam,
  type ExamRow,
  type QuestionStatus,
  type SessionSummary,
} from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'

export const Route = createFileRoute('/exam/$name')({ component: ExamView })

function ExamView() {
  const { name } = Route.useParams()
  const [exam, setExam] = useState<Exam | null>(null)
  const [rows, setRows] = useState<ExamRow[]>([])
  const [course, setCourse] = useState('')
  const [col, setCol] = useState('')
  const [index, setIndex] = useState(0)
  const [gradeValue, setGradeValue] = useState('')
  const [suggestion, setSuggestion] = useState<AIGradeSuggestion | null>(null)
  const [status, setStatus] = useState<QuestionStatus | null>(null)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [activeSession, setActiveSession] = useState('default')
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  function refreshSessions(select?: string) {
    return api
      .listSessions(name)
      .then((list) => {
        setSessions(list)
        setActiveSession((cur) => {
          if (select) return select
          if (list.some((s) => s.name === cur)) return cur
          if (list.some((s) => s.name === 'default')) return 'default'
          return list[0]?.name ?? 'default'
        })
        return list
      })
      .catch(() => {})
  }

  useEffect(() => {
    api
      .getExam(name)
      .then((s) => {
        setExam(s)
        setCourse(s.course ?? '')
        setRows([])
        setCol(s.output_columns[0] ?? '')
        return api.examRows(s.exam_file)
      })
      .then(setRows)
      .catch((e: Error) => setError(e.message))
    refreshSessions()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name])

  const question = exam && col ? exam.questions[col] : undefined
  const inputCol = question?.input_column ?? ''

  const ungraded = useMemo(() => {
    if (!question) return [] as ExamRow[]
    const idCols = exam?.id_columns ?? []
    const gradedIds = new Set(question.graded_items.map((g) => g.row_id))
    const priority = new Set(question.sampling_result?.selected_ids ?? [])
    return rows
      .filter((r) => !gradedIds.has(rowId(r, idCols)) && isMeaningful(r[inputCol] ?? ''))
      .sort(
        (a, b) =>
          Number(!priority.has(rowId(a, idCols))) - Number(!priority.has(rowId(b, idCols))),
      )
  }, [question, rows, exam, inputCol])

  const safeIndex = ungraded.length ? Math.min(index, ungraded.length - 1) : 0
  const current = ungraded[safeIndex]

  useEffect(() => {
    setIndex(0)
    setSuggestion(null)
    setGradeValue('')
  }, [col])

  useEffect(() => {
    if (col) api.questionStatus(name, col).then(setStatus).catch(() => setStatus(null))
  }, [name, col, exam])

  async function save() {
    if (!current || !col) return
    const grade = gradeValue.trim()
    if (!grade) return setError('Enter a grade (or request an AI suggestion)')
    setError(null)
    setBusy(true)
    try {
      const updated = await api.grade(
        name,
        col,
        rowId(current, exam?.id_columns ?? []),
        grade,
        activeSession,
      )
      setExam((s) => (s ? { ...s, questions: { ...s.questions, [col]: updated } } : s))
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
      const s = await api.suggest(name, col, rowId(current, exam?.id_columns ?? []))
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
      const fresh = await api.getExam(name)
      setExam(fresh)
      setInfo(`Selected ${result.num_samples} representative sample(s) via ${algorithm}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function newSession() {
    const newName = window.prompt('New session name')?.trim()
    if (!newName) return
    setError(null)
    try {
      const created = await api.createSession(name, newName)
      await refreshSessions(created.name)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function saveCourse() {
    if (!exam) return
    if ((exam.course ?? '') === course.trim()) return
    try {
      const updated = await api.updateExam(name, { course: course.trim() })
      setExam((s) => (s ? { ...s, course: updated.course } : s))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function exportExam() {
    setBusy(true)
    try {
      const { path } = await api.exportExam(name)
      setInfo(`Exported to ${path}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (!exam) {
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
      <ExamNav name={name} active="grade" />
      <h1 className="text-2xl font-bold">{exam.name}</h1>

      <label className="flex items-center gap-2 text-sm text-gray-600">
        Course:
        <input
          className="rounded border px-2 py-1"
          placeholder="e.g. CS101"
          value={course}
          onChange={(e) => setCourse(e.target.value)}
          onBlur={saveCourse}
          onKeyDown={(e) => {
            if (e.key === 'Enter') void saveCourse()
          }}
        />
      </label>

      <label className="flex items-center gap-2 text-sm text-gray-600">
        Session:
        <select
          className="rounded border px-2 py-1"
          value={activeSession}
          onChange={(e) => setActiveSession(e.target.value)}
        >
          {sessions.length === 0 && <option value="default">default (0)</option>}
          {sessions.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name} ({s.graded_count})
            </option>
          ))}
        </select>
        <button
          onClick={newSession}
          className="rounded border px-3 py-1 hover:bg-gray-50"
        >
          New session
        </button>
      </label>

      <div className="flex flex-wrap items-center gap-2">
        <select
          className="rounded border p-2"
          value={col}
          onChange={(e) => setCol(e.target.value)}
        >
          {exam.output_columns.map((c) => (
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
          onClick={exportExam}
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
            {rowId(current, exam.id_columns)}
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
