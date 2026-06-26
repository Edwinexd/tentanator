import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { api } from '#/lib/api'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from '#/components/ui/card'
import { Button } from '#/components/ui/button'
import { Input } from '#/components/ui/input'
import { Label } from '#/components/ui/label'
import { Checkbox } from '#/components/ui/checkbox'
import { Alert, AlertDescription } from '#/components/ui/alert'
import { Separator } from '#/components/ui/separator'
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '#/components/ui/select'
import { ArrowLeft, Wand2 } from 'lucide-react'

export const Route = createFileRoute('/new')({ component: NewSession })

interface ColumnPickerProps {
  title: string
  columns: string[]
  selected: Set<string>
  onToggle: (col: string) => void
}

function ColumnPicker({ title, columns, selected, onToggle }: ColumnPickerProps) {
  if (columns.length === 0) return null
  return (
    <div className="space-y-2">
      <Label className="text-base font-medium">{title}</Label>
      <div className="max-h-48 space-y-1 overflow-y-auto rounded-md border p-2">
        {columns.map((col) => (
          <label key={col} className="flex items-center gap-2 py-0.5">
            <Checkbox
              checked={selected.has(col)}
              onCheckedChange={() => onToggle(col)}
            />
            <span className="text-sm">{col}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

function NewSession() {
  const navigate = useNavigate()
  const [exams, setExams] = useState<string[]>([])
  const [examFile, setExamFile] = useState('')
  const [columns, setColumns] = useState<string[]>([])
  const [idCols, setIdCols] = useState<Set<string>>(new Set())
  const [inputCols, setInputCols] = useState<Set<string>>(new Set())
  const [outputCols, setOutputCols] = useState<Set<string>>(new Set())
  const [name, setName] = useState('')
  const [course, setCourse] = useState('')
  const [detected, setDetected] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.listExamFiles().then(setExams).catch((e: Error) => setError(e.message))
  }, [])

  function applyDetection() {
    if (!examFile) return
    api
      .detectColumns(examFile)
      .then((d) => {
        setIdCols(new Set(d.id_columns))
        setInputCols(new Set(d.input_columns))
        setOutputCols(new Set(d.output_columns))
        setDetected(d.output_columns.length)
      })
      .catch((e: Error) => setError(e.message))
  }

  useEffect(() => {
    if (!examFile) {
      setColumns([])
      setIdCols(new Set())
      setInputCols(new Set())
      setOutputCols(new Set())
      setDetected(0)
      return
    }
    api
      .examColumns(examFile)
      .then(setColumns)
      .catch((e: Error) => setError(e.message))
    applyDetection()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [examFile])

  const toggle = (set: Set<string>, setter: (s: Set<string>) => void) => (col: string) => {
    const next = new Set(set)
    if (next.has(col)) next.delete(col)
    else next.add(col)
    setter(next)
  }

  async function create() {
    setError(null)
    const id = [...idCols]
    const inp = [...inputCols]
    const out = [...outputCols]
    if (!examFile) return setError('Select an exam file')
    if (id.length === 0) return setError('Select at least one ID column')
    if (inp.length === 0) return setError('Select at least one input (response) column')
    if (out.length === 0) return setError('Select at least one output (grade) column')
    try {
      const exam = await api.createExam({
        exam_file: examFile,
        id_columns: id,
        input_columns: inp,
        output_columns: out,
        name: name || undefined,
        course: course || undefined,
      })
      navigate({ to: '/exam/$name', params: { name: exam.name } })
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-8">
      <div className="flex items-center gap-2">
        <Link to="/">
          <Button variant="ghost" size="icon" aria-label="Back">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <h1 className="text-2xl font-bold">New exam</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Exam file</CardTitle>
          <CardDescription>Choose a .xlsx or .csv file from the exams directory</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="exam-file-select">File</Label>
            <Select value={examFile} onValueChange={setExamFile}>
              <SelectTrigger id="exam-file-select">
                <SelectValue placeholder="— select —" />
              </SelectTrigger>
              <SelectContent>
                {exams.map((f) => (
                  <SelectItem key={f} value={f}>{f}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {detected > 0 && (
            <Alert>
              <AlertDescription>
                Detected {detected} question pair(s). Adjust below if needed.
                <Button
                  variant="link"
                  size="sm"
                  className="ml-2 h-auto p-0"
                  onClick={() => applyDetection()}
                >
                  <Wand2 className="mr-1 h-3 w-3" />
                  Re-detect
                </Button>
              </AlertDescription>
            </Alert>
          )}

          <Separator />

          <ColumnPicker
            title="ID columns (student identifier)"
            columns={columns}
            selected={idCols}
            onToggle={toggle(idCols, setIdCols)}
          />
          <ColumnPicker
            title="Input columns (student response text)"
            columns={columns}
            selected={inputCols}
            onToggle={toggle(inputCols, setInputCols)}
          />
          <ColumnPicker
            title="Output columns (grade / score)"
            columns={columns}
            selected={outputCols}
            onToggle={toggle(outputCols, setOutputCols)}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Exam details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="exam-name">Name (optional — auto-generated if empty)</Label>
            <Input
              id="exam-name"
              placeholder="e.g. Midterm 2026"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="exam-course">Course (optional)</Label>
            <Input
              id="exam-course"
              placeholder="e.g. CS101"
              value={course}
              onChange={(e) => setCourse(e.target.value)}
            />
          </div>
        </CardContent>
        <CardFooter className="justify-end gap-2">
          {error && (
            <Alert variant="destructive" className="flex-1">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          <Button onClick={create}>
            Create exam
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
