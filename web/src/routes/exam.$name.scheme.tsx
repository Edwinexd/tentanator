import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import {
  api,
  type QuestionConfigUpdate,
  type ResultsResponse,
} from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'
import { PageShell } from '#/components/PageShell'
import { Button } from '#/components/ui/button'
import { Input } from '#/components/ui/input'
import { Label } from '#/components/ui/label'
import { Badge } from '#/components/ui/badge'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from '#/components/ui/card'
import { Alert, AlertDescription } from '#/components/ui/alert'
import { Checkbox } from '#/components/ui/checkbox'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '#/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '#/components/ui/table'
import { Save, Eye, Play } from 'lucide-react'

export const Route = createFileRoute('/exam/$name/scheme')({ component: SchemeView })

const QTYPES: { v: string; l: string }[] = [
  { v: '', l: '—' },
  { v: 'mc', l: 'Multiple choice' },
  { v: 'sc', l: 'Single choice' },
  { v: 'essay', l: 'Essay' },
  { v: 'comment', l: 'Comment (ungraded)' },
]

function normQtype(v: string): string {
  const lo = v.toLowerCase()
  return QTYPES.some((t) => t.v === lo) ? lo : ''
}

function SchemeView() {
  const { name } = Route.useParams()
  const [cfg, setCfg] = useState<QuestionConfigUpdate[]>([])
  const [colMax, setColMax] = useState<Record<string, number>>({})
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [bulkGroup, setBulkGroup] = useState('')
  const [bulkType, setBulkType] = useState('')
  const [schemeText, setSchemeText] = useState('')
  const [preview, setPreview] = useState<ResultsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    api
      .getExam(name)
      .then(async (e) => {
        const cols = e.output_columns
        const questions = e.questions
        const cfgList: QuestionConfigUpdate[] = cols.map((col) => {
          const q = questions[col]
          const maxGuess = colMax[col] ?? undefined
          return {
            col,
            var: q?.var ?? col.replace(/\s+/g, '_').toLowerCase(),
            group: q?.group ?? '',
            qtype: normQtype(q?.qtype ?? ''),
            max_points: q?.max_points ?? maxGuess ?? 0,
            position: q?.position ?? 0,
          }
        })
        if (active) setCfg(cfgList)
        if (active) setSchemeText(e.scheme ? await api.schemeEmit(e.scheme) : '')
        return api.examRows(e.exam_file)
      })
      .then((rows) => {
        const m: Record<string, number> = {}
        for (const col of Object.keys(rows[0] ?? {})) {
          let max = 0
          for (const r of rows) {
            const v = parseFloat(r[col])
            if (!isNaN(v) && v > max) max = v
          }
          if (max > 0) m[col] = max
        }
        if (active) setColMax(m)
      })
      .catch((e: Error) => { if (active) setError(e.message) })
    return () => { active = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name])

  function setRow(i: number, patch: Partial<QuestionConfigUpdate>) {
    setCfg((rows) => rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  }

  function applyToSelected(patch: (r: QuestionConfigUpdate) => Partial<QuestionConfigUpdate>) {
    setCfg((rows) =>
      rows.map((r) => (selected.size === 0 || selected.has(r.col) ? { ...r, ...patch(r) } : r)),
    )
  }

  function toggleSel(col: string) {
    setSelected((s) => {
      const n = new Set(s)
      if (n.has(col)) n.delete(col); else n.add(col)
      return n
    })
  }

  function toggleAll() {
    setSelected((s) => (s.size === cfg.length ? new Set() : new Set(cfg.map((r) => r.col))))
  }


  async function saveConfig() {
    setError(null)
    try {
      await api.putQuestionsConfig(name, cfg)
      setInfo('Question config saved')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function doPreview() {
    setError(null)
    setPreview(null)
    try {
      const s = await api.schemeParse(schemeText)
      await api.putQuestionsConfig(name, cfg)
      const r = await api.previewResults(name, s)
      setPreview(r)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function saveScheme() {
    setError(null)
    try {
      const s = await api.schemeParse(schemeText)
      await api.putQuestionsConfig(name, cfg)
      await api.putScheme(name, s)
      setInfo('Scheme saved')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const scope = selected.size === 0 ? 'all' : `${selected.size} selected`

  return (
    <PageShell>
      <ExamNav name={name} active="scheme" />
      <h1 className="text-2xl font-bold">Grade scheme</h1>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            Question config
            <Badge variant="secondary" className="text-xs">{scope}</Badge>
          </CardTitle>
          <CardDescription>Configure each question's type, max points, and group</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Label className="text-sm">Bulk set group:</Label>
            <Input
              className="h-8 w-40"
              placeholder="group name"
              value={bulkGroup}
              onChange={(e) => setBulkGroup(e.target.value)}
            />
            <Button
              size="sm"
              variant="outline"
              onClick={() => applyToSelected(() => ({ group: bulkGroup }))}
            >
              Apply
            </Button>

            <Label className="ml-2 text-sm">Bulk set type:</Label>
            <Select value={bulkType} onValueChange={setBulkType}>
              <SelectTrigger className="h-8 w-auto">
                <SelectValue placeholder="—" />
              </SelectTrigger>
              <SelectContent>
                {QTYPES.map((t) => <SelectItem key={t.v} value={t.v}>{t.l}</SelectItem>)}
              </SelectContent>
            </Select>
            <Button
              size="sm"
              variant="outline"
              onClick={() => applyToSelected(() => ({ qtype: bulkType }))}
            >
              Apply
            </Button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left">
                  <th className="p-2">
                    <Checkbox
                      checked={selected.size === cfg.length && cfg.length > 0}
                      onCheckedChange={toggleAll}
                    />
                  </th>
                  <th className="p-2 font-medium">Column</th>
                  <th className="p-2 font-medium">Var</th>
                  <th className="p-2 font-medium">Group</th>
                  <th className="p-2 font-medium">Type</th>
                  <th className="p-2 font-medium">Max</th>
                  <th className="p-2 font-medium">Pos</th>
                </tr>
              </thead>
              <tbody>
                {cfg.map((r, i) => (
                  <tr key={r.col} className="border-b hover:bg-muted/50">
                    <td className="p-2">
                      <Checkbox
                        checked={selected.has(r.col)}
                        onCheckedChange={() => toggleSel(r.col)}
                      />
                    </td>
                    <td className="p-2 text-muted-foreground">{r.col}</td>
                    <td className="p-2">
                      <Input
                        className="h-7 w-32"
                        value={r.var ?? ''}
                        onChange={(e) => setRow(i, { var: e.target.value })}
                      />
                    </td>
                    <td className="p-2">
                      <Input
                        className="h-7 w-24"
                        value={r.group ?? ''}
                        onChange={(e) => setRow(i, { group: e.target.value })}
                      />
                    </td>
                    <td className="p-2">
                      <select
                        className="h-7 rounded-md border border-input bg-transparent px-2 text-sm"
                        value={r.qtype ?? ''}
                        onChange={(e) => setRow(i, { qtype: e.target.value })}
                      >
                        {QTYPES.map((t) => <option key={t.v} value={t.v}>{t.l}</option>)}
                      </select>
                    </td>
                    <td className="p-2">
                      <Input
                        className="h-7 w-16"
                        type="number"
                        value={r.max_points ?? ''}
                        onChange={(e) => setRow(i, { max_points: e.target.value ? Number(e.target.value) : 0 })}
                      />
                    </td>
                    <td className="p-2">
                      <Input
                        className="h-7 w-16"
                        type="number"
                        value={r.position ?? ''}
                        onChange={(e) => setRow(i, { position: e.target.value ? Number(e.target.value) : 0 })}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
        <CardFooter>
          <Button onClick={saveConfig} variant="outline" size="sm">
            <Save className="mr-1 h-4 w-4" />
            Save config
          </Button>
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Grade scheme</CardTitle>
          <CardDescription>
            One statement per line: <code>const name = value</code>,{' '}
            <code>name = expr</code>, <code>when cond -&gt; grade</code>. Optional{' '}
            <code>total_var:</code> / <code>default_grade:</code>. Blank lines and{' '}
            <code>#</code> comments ignored.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <textarea
            className="min-h-[200px] w-full rounded-md border border-input bg-transparent p-3 font-mono text-sm"
            value={schemeText}
            onChange={(e) => setSchemeText(e.target.value)}
          />
          <div className="flex gap-2">
            <Button onClick={doPreview} variant="outline" size="sm">
              <Eye className="mr-1 h-4 w-4" />
              Preview
            </Button>
            <Button onClick={saveScheme} size="sm">
              <Save className="mr-1 h-4 w-4" />
              Save scheme
            </Button>
          </div>
        </CardContent>
      </Card>

      {preview && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Play className="h-5 w-5 text-green-600" />
              Results preview
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary">{preview.results.length} students</Badge>
              <Badge variant="secondary">{preview.unresolved_conflicts} conflict(s)</Badge>
            </div>
            <div className="max-h-80 overflow-auto rounded border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead>Grade</TableHead>
                    <TableHead>Total</TableHead>
                    <TableHead>Estimate</TableHead>
                    <TableHead>Complete</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {preview.results.map((s) => (
                    <TableRow key={s.id}>
                      <TableCell className="font-mono text-xs">{s.id}</TableCell>
                      <TableCell>{s.grade || '—'}</TableCell>
                      <TableCell>{s.total.toFixed(1)}</TableCell>
                      <TableCell>{s.estimated.length > 0 ? s.estimated.join(', ') : '—'}</TableCell>
                      <TableCell>{s.complete ? '✓' : '…'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {info && <Alert><AlertDescription>{info}</AlertDescription></Alert>}
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
    </PageShell>
  )
}
