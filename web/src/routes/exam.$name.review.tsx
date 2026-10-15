import { createFileRoute } from '@tanstack/react-router'
import { useCallback, useEffect, useState, type ChangeEvent } from 'react'
import {
  api,
  type SecondPassImport,
  type SecondPassQuestion,
  type SecondPassReport,
  type SecondPassStudent,
} from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'
import { PageShell } from '#/components/PageShell'
import { Button } from '#/components/ui/button'
import { Badge } from '#/components/ui/badge'
import { Input } from '#/components/ui/input'
import { Label } from '#/components/ui/label'
import { Alert, AlertDescription } from '#/components/ui/alert'
import { Card, CardHeader, CardTitle, CardContent } from '#/components/ui/card'
import { ChevronDown, ChevronRight, Download, Upload } from 'lucide-react'

export const Route = createFileRoute('/exam/$name/review')({ component: ReviewView })

const fmt = (n: number) => String(Math.round(n * 100) / 100)

function ReviewView() {
  const { name } = Route.useParams()
  const [margin, setMargin] = useState(1)
  const [data, setData] = useState<SecondPassReport | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const [file, setFile] = useState('')
  const [plan, setPlan] = useState<SecondPassImport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)

  const load = useCallback(() => {
    if (!(margin > 0)) return
    api
      .secondPass(name, margin)
      .then(setData)
      .catch((e: Error) => setError(e.message))
  }, [name, margin])

  useEffect(() => { load() }, [load])

  async function run(fn: () => Promise<void>) {
    setError(null)
    setInfo(null)
    try {
      await fn()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  function onUpload(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    e.target.value = ''
    if (!f) return
    void run(async () => {
      const { filename } = await api.uploadExamFile(f)
      setFile(filename)
      setPlan(await api.secondPassImport(name, { file: filename, dry_run: true }))
    })
  }

  function apply() {
    void run(async () => {
      const res = await api.secondPassImport(name, { file })
      setPlan(null)
      setFile('')
      setInfo(
        `Imported ${res.changed.length} change(s); ${res.grade_changes.length} final grade(s) moved`,
      )
      load()
    })
  }

  const transitions: Record<string, number> = {}
  for (const s of data?.students ?? []) {
    const k = `${s.grade}→${s.gap.next_grade}`
    transitions[k] = (transitions[k] ?? 0) + 1
  }

  return (
    <PageShell>
      <ExamNav name={name} active="review" />
      <h1 className="text-2xl font-bold">Second pass</h1>
      <p className="text-sm text-muted-foreground">
        Students within a margin of a better grade, with the answers where they can still gain
        points.
      </p>

      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <Label htmlFor="margin">Margin (points)</Label>
          <Input
            id="margin"
            type="number"
            min={0.25}
            step={0.25}
            value={margin}
            onChange={(e) => setMargin(Number(e.target.value))}
            className="w-28"
          />
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => run(async () => {
            await api.exportSecondPass(name, margin)
            setInfo('Download started')
          })}
        >
          <Download className="mr-1 h-4 w-4" />
          Export review sheet
        </Button>
        <span className="text-xs text-muted-foreground">
          Fill the yellow <em>New points</em> column, then import it below.
        </span>
      </div>

      {info && <Alert><AlertDescription>{info}</AlertDescription></Alert>}
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Import review sheet</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <label className="inline-flex cursor-pointer items-center gap-2 text-sm">
            <Upload className="h-4 w-4" />
            <span className="underline">Choose filled sheet…</span>
            <input type="file" accept=".xlsx,.csv" className="hidden" onChange={onUpload} />
          </label>
          {plan && <ImportPlan plan={plan} file={file} onApply={apply} />}
        </CardContent>
      </Card>

      {!data && !error && <p className="text-muted-foreground">Loading…</p>}
      {data && !data.has_scheme && (
        <p className="text-muted-foreground">
          No grade scheme yet. Configure one on the Scheme tab first.
        </p>
      )}

      {data && data.has_scheme && (
        <>
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">
              {data.students.length} student(s) within {fmt(data.margin)} pt
            </Badge>
            {Object.entries(transitions).map(([k, n]) => (
              <Badge key={k} variant="outline">{k}: {n}</Badge>
            ))}
          </div>
          <div className="space-y-2">
            {data.students.map((s) => (
              <StudentRow
                key={s.id}
                exam={name}
                student={s}
                open={open === s.id}
                onToggle={() => setOpen(open === s.id ? null : s.id)}
                onSaved={load}
                onError={setError}
              />
            ))}
          </div>
        </>
      )}
    </PageShell>
  )
}

function ImportPlan({
  plan,
  file,
  onApply,
}: {
  plan: SecondPassImport
  file: string
  onApply: () => void
}) {
  return (
    <div className="space-y-2 text-sm">
      <div className="flex flex-wrap gap-2">
        <Badge variant="secondary">{file}</Badge>
        <Badge variant="secondary">{plan.changed.length} changed</Badge>
        <Badge variant="secondary">{plan.unchanged} unchanged</Badge>
        <Badge variant="secondary">{plan.blank} blank</Badge>
        {plan.invalid.length > 0 && (
          <Badge variant="destructive">{plan.invalid.length} invalid</Badge>
        )}
      </div>
      {plan.changed.length > 0 && (
        <ul className="font-mono text-xs">
          {plan.changed.map((c) => (
            <li key={`${c.id}/${c.col}`}>
              {c.id} {c.col}: {c.before || '(none)'} → {c.after}
            </li>
          ))}
        </ul>
      )}
      {plan.grade_changes.length > 0 && (
        <div>
          <div className="font-semibold">Final grade changes</div>
          <ul className="font-mono text-xs">
            {plan.grade_changes.map((g) => (
              <li key={g.id}>{g.id}: {g.before} → {g.after}</li>
            ))}
          </ul>
        </div>
      )}
      {plan.invalid.length > 0 && (
        <ul className="text-xs text-destructive">
          {plan.invalid.map((m) => <li key={m}>{m}</li>)}
        </ul>
      )}
      <Button size="sm" onClick={onApply} disabled={plan.changed.length === 0}>
        Apply {plan.changed.length} change(s)
      </Button>
    </div>
  )
}

function StudentRow({
  exam,
  student: s,
  open,
  onToggle,
  onSaved,
  onError,
}: {
  exam: string
  student: SecondPassStudent
  open: boolean
  onToggle: () => void
  onSaved: () => void
  onError: (m: string) => void
}) {
  return (
    <Card>
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full flex-wrap items-center gap-3 px-4 py-2 text-left"
      >
        {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        <span className="font-mono text-xs">{s.id}</span>
        <span className="font-semibold">{s.grade} → {s.gap.next_grade}</span>
        <span className="text-sm text-muted-foreground">total {fmt(s.total)}</span>
        <Badge variant="outline">+{fmt(s.gap.points)}</Badge>
        {!s.complete && <Badge variant="destructive">incomplete</Badge>}
      </button>
      {open && (
        <CardContent className="space-y-3">
          {s.questions.map((q) => (
            <QuestionReview
              key={q.col}
              exam={exam}
              rowId={s.id}
              q={q}
              liftGrade={
                s.gap.lifts.find((l) => l.source === 'question' && l.key === q.col)?.grade ??
                s.gap.next_grade
              }
              onSaved={onSaved}
              onError={onError}
            />
          ))}
        </CardContent>
      )}
    </Card>
  )
}

function QuestionReview({
  exam,
  rowId,
  q,
  liftGrade,
  onSaved,
  onError,
}: {
  exam: string
  rowId: string
  q: SecondPassQuestion
  liftGrade: string
  onSaved: () => void
  onError: (m: string) => void
}) {
  const [value, setValue] = useState(q.points == null ? '' : String(q.points))
  const [saving, setSaving] = useState(false)

  async function save() {
    setSaving(true)
    try {
      await api.grade(exam, q.col, rowId, value.trim())
      onSaved()
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-1 border-t pt-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">{q.label}</span>
        {q.group && <Badge variant="secondary">{q.group}</Badge>}
        <span>{q.points == null ? 'ungraded' : fmt(q.points)} / {fmt(q.max)}</span>
        {q.lift != null && <Badge>+{fmt(q.lift)} alone → {liftGrade}</Badge>}
      </div>
      <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded bg-muted p-2 text-xs">
        {q.response || '(no response)'}
      </pre>
      <div className="flex items-center gap-2">
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void save() }}
          className="w-28"
          aria-label={`${q.label} points`}
        />
        <Button
          size="sm"
          variant="outline"
          onClick={() => void save()}
          disabled={saving || value.trim() === ''}
        >
          Save
        </Button>
      </div>
    </div>
  )
}
