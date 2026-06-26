import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  api,
  detectQuestionPairs,
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
import { Button } from '#/components/ui/button'
import { Input } from '#/components/ui/input'
import { Textarea } from '#/components/ui/textarea'
import { Label } from '#/components/ui/label'
import { Checkbox } from '#/components/ui/checkbox'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '#/components/ui/select'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '#/components/ui/accordion'
import {
  Card,
  CardContent,
} from '#/components/ui/card'
import { Alert, AlertDescription } from '#/components/ui/alert'
import {
  Wand2,
  Shuffle,
  Download,
  Plus,
  Sparkles,
  SkipForward,
  Save,
  Brain,
} from 'lucide-react'

export const Route = createFileRoute('/exam/$name/')({ component: ExamView })

function ExamView() {
  const { name } = Route.useParams()
  const [exam, setExam] = useState<Exam | null>(null)
  const [rows, setRows] = useState<ExamRow[]>([])
  const [fileColumns, setFileColumns] = useState<string[]>([])
  const [course, setCourse] = useState('')
  const [col, setCol] = useState('')
  const didInitCol = useRef(false)
  const [index, setIndex] = useState(0)
  const [gradeValue, setGradeValue] = useState('')
  const [suggestions, setSuggestions] = useState<Record<string, AIGradeSuggestion>>({})
  const [pending, setPending] = useState<Record<string, boolean>>({})
  const [status, setStatus] = useState<QuestionStatus | null>(null)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [activeSession, setActiveSession] = useState('default')
  const [autoSuggest, setAutoSuggest] = useState(true)
  const [qGlobalId, setQGlobalId] = useState('')
  const [qExamText, setQExamText] = useState('')
  const [qSample, setQSample] = useState('')
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
    didInitCol.current = false
    api
      .getExam(name)
      .then((s) => {
        setExam(s)
        setCourse(s.course ?? '')
        setRows([])
        setFileColumns([])
        api
          .examColumns(s.exam_file)
          .then(setFileColumns)
          .catch(() => setFileColumns([]))
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

  const colStatus = useMemo(() => {
    const map: Record<string, { applicable: number; graded: number; ungraded: number }> = {}
    if (!exam) return map
    for (const c of exam.output_columns) {
      const q = exam.questions[c]
      const applicable = rows.filter((r) => isMeaningful(r[q?.input_column ?? ''] ?? '')).length
      const graded = q?.graded_items.length ?? 0
      map[c] = { applicable, graded, ungraded: applicable - graded }
    }
    return map
  }, [exam, rows])

  useEffect(() => {
    if (didInitCol.current || !exam || rows.length === 0) return
    if (exam.output_columns.length === 0) {
      didInitCol.current = true
      return
    }
    const needy = exam.output_columns.find((c) => (colStatus[c]?.ungraded ?? 0) > 0)
    setCol(needy ?? exam.output_columns[0])
    didInitCol.current = true
  }, [exam, rows, colStatus])

  useEffect(() => { setIndex(0) }, [col])

  useEffect(() => {
    const q = exam?.questions[col]
    setQGlobalId(q?.global_question_id ?? '')
    setQExamText(q?.exam_question ?? '')
    setQSample(q?.sample_answer ?? '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [col])

  useEffect(() => {
    if (col) api.questionStatus(name, col).then(setStatus).catch(() => setStatus(null))
  }, [name, col, exam])

  const idCols = exam?.id_columns ?? []
  const suggestKey = (c: string, r: ExamRow) => `${c}::${rowId(r, idCols)}`
  const iclReady = status?.icl_ready ?? false

  async function fetchSuggestion(
    targetCol: string,
    row: ExamRow,
  ): Promise<AIGradeSuggestion | undefined> {
    if (!iclReady) return undefined
    const key = suggestKey(targetCol, row)
    if (suggestions[key]) return suggestions[key]
    if (pending[key]) return undefined
    setPending((p) => ({ ...p, [key]: true }))
    try {
      const s = await api.suggest(name, targetCol, rowId(row, idCols))
      setSuggestions((m) => ({ ...m, [key]: s }))
      return s
    } catch {
      return undefined
    } finally {
      setPending((p) => {
        const n = { ...p }
        delete n[key]
        return n
      })
    }
  }

  useEffect(() => {
    if (!autoSuggest || !current || !col || !iclReady) return
    void fetchSuggestion(col, current)
    const next = ungraded[safeIndex + 1]
    if (next) void fetchSuggestion(col, next)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoSuggest, current, col, iclReady, ungraded, safeIndex])

  const curKey = current ? suggestKey(col, current) : ''
  const curSuggestion = curKey ? suggestions[curKey] : undefined

  useEffect(() => {
    setGradeValue(autoSuggest && curSuggestion ? curSuggestion.grade : '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [curKey])

  useEffect(() => {
    if (autoSuggest && curSuggestion && gradeValue === '') setGradeValue(curSuggestion.grade)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [curSuggestion])

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
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function suggestNow() {
    if (!current || !col) return
    const s = await fetchSuggestion(col, current)
    if (s) setGradeValue(s.grade)
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

  async function saveQuestion() {
    if (!col) return
    setError(null)
    try {
      const updated = await api.putQuestion(name, col, {
        exam_question: qExamText,
        sample_answer: qSample,
        global_question_id: qGlobalId.trim(),
      })
      setExam((s) => (s ? { ...s, questions: { ...s.questions, [col]: updated } } : s))
      setInfo(
        qGlobalId.trim()
          ? `Linked to "${qGlobalId.trim()}" - prior grades for this id are now shared examples`
          : 'Question settings saved',
      )
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function exportExam() {
    setBusy(true)
    try {
      await api.exportExam(name)
      setInfo('Download started')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function addQuestions() {
    if (!exam) return
    setBusy(true)
    setInfo(null)
    setError(null)
    try {
      const det = detectQuestionPairs(fileColumns)
      const extraOut = exam.output_columns.filter((c) => !det.output_columns.includes(c))
      const output_columns = [...det.output_columns, ...extraOut]
      const input_columns = [
        ...det.input_columns,
        ...extraOut.map((c) => exam.questions[c]?.input_column ?? ''),
      ]
      const id_columns = exam.id_columns.length ? exam.id_columns : det.id_columns
      const updated = await api.updateExamColumns(name, {
        id_columns,
        input_columns,
        output_columns,
      })
      setExam(updated)
      const fresh = await api.examRows(updated.exam_file)
      setRows(fresh)
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
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : (
          <p className="text-muted-foreground">Loading…</p>
        )}
        <Link to="/">
          <Button variant="link" className="mt-4">← back</Button>
        </Link>
      </div>
    )
  }

  const gradedCount = question?.graded_items.length ?? 0
  const iclHave = (status?.valid_graded ?? 0) + (status?.external ?? 0)
  const iclNeed = Math.max(0, (status?.min_icl_examples ?? 5) - iclHave)
  const det = detectQuestionPairs(fileColumns)
  const missing = det.output_columns.filter((c) => !exam.output_columns.includes(c))

  return (
    <div className="mx-auto max-w-3xl space-y-4 p-8">
      <ExamNav name={name} active="grade" />
      <h1 className="text-2xl font-bold">{exam.name}</h1>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <Label htmlFor="course-input" className="text-sm text-muted-foreground">Course:</Label>
          <Input
            id="course-input"
            className="h-8 w-40"
            placeholder="e.g. CS101"
            value={course}
            onChange={(e) => setCourse(e.target.value)}
            onBlur={saveCourse}
            onKeyDown={(e) => { if (e.key === 'Enter') void saveCourse() }}
          />
        </div>

        <div className="flex items-center gap-2">
          <Label htmlFor="session-select" className="text-sm text-muted-foreground">Session:</Label>
          <Select value={activeSession} onValueChange={setActiveSession}>
            <SelectTrigger className="h-8 w-auto min-w-[8rem]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {sessions.length === 0 && <SelectItem value="default">default (0)</SelectItem>}
              {sessions.map((s) => (
                <SelectItem key={s.name} value={s.name}>
                  {s.name} ({s.graded_count})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button onClick={newSession} variant="outline" size="sm">
            <Plus className="h-3 w-3" />
            New session
          </Button>
        </div>

        <label className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <Checkbox
            checked={autoSuggest}
            onCheckedChange={(c) => setAutoSuggest(c === true)}
          />
          <span>auto AI-suggest</span>
        </label>
      </div>

      {missing.length > 0 && (
        <Alert>
          <AlertDescription className="flex flex-wrap items-center gap-3">
            <span>
              This file has {det.output_columns.length} question(s); this exam covers{' '}
              {exam.output_columns.length}.
            </span>
            <Button
              disabled={busy}
              onClick={addQuestions}
              variant="secondary"
              size="sm"
            >
              <Wand2 className="mr-1 h-3 w-3" />
              Add {missing.length} more
            </Button>
          </AlertDescription>
        </Alert>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Select value={col} onValueChange={setCol}>
          <SelectTrigger className="w-auto min-w-[12rem]">
            <SelectValue placeholder="Select question" />
          </SelectTrigger>
          <SelectContent>
            {exam.output_columns.map((c) => (
              <SelectItem key={c} value={c}>
                {c} ({colStatus[c]?.graded ?? 0}/{colStatus[c]?.applicable ?? 0})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button disabled={busy} onClick={() => sample('random')} variant="outline" size="sm">
          <Shuffle className="mr-1 h-3 w-3" />
          random sample
        </Button>
        <Button disabled={busy} onClick={() => sample('maximin')} variant="outline" size="sm">
          <Wand2 className="mr-1 h-3 w-3" />
          maximin sample
        </Button>
        <Button disabled={busy} onClick={exportExam} variant="outline" size="sm">
          <Download className="mr-1 h-3 w-3" />
          export
        </Button>
      </div>

      <Accordion type="single" collapsible>
        <AccordionItem value="question-settings">
          <AccordionTrigger className="text-sm text-muted-foreground">
            Question settings
            {qGlobalId.trim() ? ` · linked: ${qGlobalId.trim()}` : ''}
            {status ? ` · ${status.external} pooled example(s)` : ''}
          </AccordionTrigger>
          <AccordionContent className="space-y-3">
            <div className="space-y-2">
              <Label htmlFor="global-id">
                Global question id — link the same question across exams/terms to share graded examples
              </Label>
              <Input
                id="global-id"
                placeholder="e.g. pvt_q37_version_control"
                value={qGlobalId}
                onChange={(e) => setQGlobalId(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="exam-text">Exam question text</Label>
              <Textarea
                id="exam-text"
                rows={2}
                value={qExamText}
                onChange={(e) => setQExamText(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="sample-answer">Sample answer (optional)</Label>
              <Textarea
                id="sample-answer"
                rows={2}
                value={qSample}
                onChange={(e) => setQSample(e.target.value)}
              />
            </div>
            <Button onClick={saveQuestion} variant="outline" size="sm">
              Save question settings
            </Button>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      <div className="text-sm text-muted-foreground">
        graded {gradedCount}/{rows.length} · {ungraded.length} ungraded ·{' '}
        ICL ready: {status ? (status.icl_ready ? 'yes' : 'no') : '…'} ·{' '}
        {status?.external ?? 0} pooled
      </div>

      {info && <Alert><AlertDescription>{info}</AlertDescription></Alert>}
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}

      {current ? (
        <>
          <div className="text-sm text-muted-foreground">
            {safeIndex + 1}/{ungraded.length} ungraded · id: {rowId(current, exam.id_columns)}
          </div>

          <Card>
            <CardContent className="max-h-80 overflow-auto whitespace-pre-wrap p-4">
              {current[inputCol]}
            </CardContent>
          </Card>

          <div className="min-h-[4.75rem] rounded-md border-l-4 border-primary bg-muted/50 p-3 text-sm">
            {curSuggestion ? (
              <div className="space-y-1">
                <div className="flex items-center gap-1 font-medium">
                  <Brain className="h-4 w-4" />
                  AI grade: {curSuggestion.grade}
                </div>
                {curSuggestion.reasoning_summary && (
                  <div className="text-muted-foreground">{curSuggestion.reasoning_summary}</div>
                )}
              </div>
            ) : pending[curKey] ? (
              <span className="text-muted-foreground">
                <Sparkles className="mr-1 inline h-4 w-4 animate-pulse" />
                Thinking…
              </span>
            ) : !iclReady ? (
              <span className="text-muted-foreground">
                AI suggestions unlock after {status?.min_icl_examples ?? 5} graded examples
                {' '}({iclHave} so far{iclNeed ? `, ${iclNeed} to go` : ''}).
              </span>
            ) : (
              <span className="text-muted-foreground">No suggestion for this response.</span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Input
              className="flex-1"
              placeholder="grade e.g. 7.5 or 2+1.5"
              value={gradeValue}
              onChange={(e) => setGradeValue(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') void save() }}
            />
            <Button
              disabled={busy || !iclReady || !!pending[curKey]}
              onClick={suggestNow}
              variant="outline"
              title={!iclReady ? `Needs ${iclNeed} more graded example(s)` : undefined}
            >
              {pending[curKey] ? (
                <><Sparkles className="mr-1 h-4 w-4 animate-pulse" />Thinking…</>
              ) : (
                <><Brain className="mr-1 h-4 w-4" />AI suggest</>
              )}
            </Button>
            <Button disabled={busy} onClick={save}>
              <Save className="mr-1 h-4 w-4" />
              Save
            </Button>
            <Button
              disabled={busy || ungraded.length < 2}
              onClick={() => setIndex((i) => (i + 1) % ungraded.length)}
              variant="outline"
            >
              <SkipForward className="mr-1 h-4 w-4" />
              Skip
            </Button>
          </div>
        </>
      ) : (
        <Card>
          <CardContent className="p-6 text-center text-muted-foreground">
            Nothing left to grade for this question.
          </CardContent>
        </Card>
      )}
    </div>
  )
}
