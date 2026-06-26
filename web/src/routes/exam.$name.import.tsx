import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState, type ChangeEvent } from 'react'
import { api, type Exam, type GradeConflict, type ImportSummary } from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'
import { Button } from '#/components/ui/button'
import { Label } from '#/components/ui/label'
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '#/components/ui/card'
import { Badge } from '#/components/ui/badge'
import { Alert, AlertDescription } from '#/components/ui/alert'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '#/components/ui/select'
import { Upload, CheckCircle2, AlertTriangle } from 'lucide-react'

export const Route = createFileRoute('/exam/$name/import')({ component: ImportView })

function ImportView() {
  const { name } = Route.useParams()
  const [exam, setExam] = useState<Exam | null>(null)
  const [exams, setExams] = useState<string[]>([])
  const [file, setFile] = useState('')
  const [columns, setColumns] = useState<string[]>([])
  const [idColumn, setIdColumn] = useState('')
  const [mapping, setMapping] = useState<Record<string, string>>({})
  const [summary, setSummary] = useState<ImportSummary | null>(null)
  const [conflicts, setConflicts] = useState<GradeConflict[]>([])
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)

  function loadConflicts() {
    api.getConflicts(name).then(setConflicts).catch(() => {})
  }

  useEffect(() => {
    api.getExam(name).then(setExam).catch((e: Error) => setError(e.message))
    api.listExamFiles().then(setExams).catch(() => {})
    loadConflicts()
  }, [name])

  useEffect(() => {
    if (!file) { setColumns([]); return }
    api.examColumns(file).then(setColumns).catch((e: Error) => setError(e.message))
  }, [file])

  async function onUpload(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setError(null)
    try {
      const path = await api.uploadExamFile(f)
      setFile(path)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  function buildReq() {
    const mappingEntries = Object.entries(mapping).filter(([, v]) => v && v !== '_skip')
    return {
      file,
      id_column: idColumn,
      column_mapping: mappingEntries.map(([from, to]) => ({ from, to })),
    }
  }

  async function preview() {
    setError(null)
    setSummary(null)
    try {
      const r = await api.importPreview(name, buildReq())
      setSummary(r)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function apply() {
    setError(null)
    try {
      const r = await api.importApply(name, buildReq())
      setInfo(`Imported ${r.imported} cell(s) (${r.new_rows} new, ${r.updated_rows} updated)`)
      setSummary(null)
      loadConflicts()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function resolve(c: GradeConflict, choose: 'existing' | 'incoming') {
    try {
      await api.resolveConflict(name, c.row_id, c.column, choose)
      loadConflicts()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const outCols = exam?.output_columns ?? []
  return (
    <div className="mx-auto max-w-4xl space-y-5 p-8">
      <ExamNav name={name} active="import" />
      <h1 className="text-2xl font-bold">Import grades</h1>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Upload a graded spreadsheet</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Select an existing exam file</Label>
            <Select value={file} onValueChange={setFile}>
              <SelectTrigger>
                <SelectValue placeholder="Choose a file in exams/" />
              </SelectTrigger>
              <SelectContent>
                {exams.map((f) => (
                  <SelectItem key={f} value={f}>{f}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>Or upload a new one:</span>
            <input
              type="file"
              accept=".xlsx,.csv"
              onChange={onUpload}
              className="text-sm file:mr-2 file:rounded file:border-0 file:bg-primary file:px-3 file:py-1 file:text-xs file:text-primary-foreground"
            />
          </div>
        </CardContent>
      </Card>

      {file && columns.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Column mapping</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>ID column in import file</Label>
              <Select value={idColumn} onValueChange={setIdColumn}>
                <SelectTrigger className="w-64">
                  <SelectValue placeholder="Select ID column" />
                </SelectTrigger>
                <SelectContent>
                  {columns.map((c) => (
                    <SelectItem key={c} value={c}>{c}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Map import columns to exam questions</Label>
              <div className="space-y-2">
                {outCols.map((outCol) => (
                  <div key={outCol} className="flex items-center gap-2">
                    <span className="w-48 text-sm font-medium">{outCol}</span>
                    <Select
                      value={mapping[outCol] ?? ''}
                      onValueChange={(v) => setMapping((m) => ({ ...m, [outCol]: v }))}
                    >
                      <SelectTrigger className="w-64">
                        <SelectValue placeholder="— skip —" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="_skip">— skip —</SelectItem>
                        {columns.map((c) => (
                          <SelectItem key={c} value={c}>{c}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                ))}
              </div>
            </div>

            <Button onClick={preview} disabled={!idColumn}>
              <Upload className="mr-1 h-4 w-4" />
              Preview import
            </Button>
          </CardContent>
        </Card>
      )}

      {summary && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <CheckCircle2 className="h-5 w-5 text-green-600" />
              Preview: {summary.new_cells} new, {summary.update_cells} updates
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-3 flex gap-2">
              <Badge variant="secondary">{summary.new_rows} new rows</Badge>
              <Badge variant="secondary">{summary.updated_rows} updated rows</Badge>
              <Badge variant="secondary">{summary.conflicts} conflict(s)</Badge>
            </div>
            <Button onClick={apply}>Apply import</Button>
          </CardContent>
        </Card>
      )}

      {conflicts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
              {conflicts.length} unresolved conflict(s)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {conflicts.map((c) => (
              <div key={`${c.row_id}_${c.column}`} className="flex items-center justify-between rounded border p-2 text-sm">
                <div>
                  <span className="font-medium">{c.row_id}</span> / <span>{c.column}</span>:{' '}
                  existing=<span className="font-medium">{c.existing}</span>{' '}
                  incoming=<span className="font-medium">{c.incoming}</span>
                </div>
                <div className="flex gap-1">
                  <Button onClick={() => resolve(c, 'existing')} variant="outline" size="sm">
                    Keep {c.existing}
                  </Button>
                  <Button onClick={() => resolve(c, 'incoming')} variant="outline" size="sm">
                    Use {c.incoming}
                  </Button>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {info && <Alert><AlertDescription>{info}</AlertDescription></Alert>}
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
    </div>
  )
}
