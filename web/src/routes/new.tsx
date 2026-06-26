import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { api } from '#/lib/api'

export const Route = createFileRoute('/new')({ component: NewSession })

function ColumnPicker({
  title,
  columns,
  selected,
  onToggle,
}: {
  title: string
  columns: string[]
  selected: Set<string>
  onToggle: (col: string) => void
}) {
  return (
    <div>
      <h3 className="mb-1 font-medium">{title}</h3>
      <div className="flex flex-wrap gap-2 rounded border p-2">
        {columns.map((c) => (
          <label
            key={c}
            className={`cursor-pointer rounded px-2 py-1 text-sm ${
              selected.has(c) ? 'bg-blue-600 text-white' : 'bg-gray-100'
            }`}
          >
            <input
              type="checkbox"
              className="mr-1"
              checked={selected.has(c)}
              onChange={() => onToggle(c)}
            />
            {c}
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
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.listExamFiles().then(setExams).catch((e: Error) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!examFile) return
    setColumns([])
    setIdCols(new Set())
    setInputCols(new Set())
    setOutputCols(new Set())
    api.examColumns(examFile).then(setColumns).catch((e: Error) => setError(e.message))
  }, [examFile])

  const toggle = (set: Set<string>, setter: (s: Set<string>) => void) => (col: string) => {
    const next = new Set(set)
    if (next.has(col)) next.delete(col)
    else next.add(col)
    setter(next)
  }

  async function create() {
    setError(null)
    if (!examFile) return setError('Pick an exam file')
    if (inputCols.size === 0 || outputCols.size === 0)
      return setError('Select at least one input and one output column')
    try {
      const exam = await api.createExam({
        exam_file: examFile,
        id_columns: [...idCols],
        input_columns: [...inputCols],
        output_columns: [...outputCols],
        name: name.trim() || undefined,
        course: course.trim() || undefined,
      })
      navigate({ to: '/exam/$name', params: { name: exam.name } })
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-4 p-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">New exam</h1>
        <Link to="/" className="text-blue-600 hover:underline">
          ← back
        </Link>
      </div>

      {error && <p className="rounded bg-red-100 p-3 text-red-700">{error}</p>}

      <div>
        <h3 className="mb-1 font-medium">Exam file</h3>
        <select
          className="w-full rounded border p-2"
          value={examFile}
          onChange={(e) => setExamFile(e.target.value)}
        >
          <option value="">Select an exam file…</option>
          {exams.map((e) => (
            <option key={e} value={e}>
              {e}
            </option>
          ))}
        </select>
      </div>

      {columns.length > 0 && (
        <>
          <ColumnPicker
            title="ID columns (student identifier)"
            columns={columns}
            selected={idCols}
            onToggle={toggle(idCols, setIdCols)}
          />
          <ColumnPicker
            title="Input columns (student responses)"
            columns={columns}
            selected={inputCols}
            onToggle={toggle(inputCols, setInputCols)}
          />
          <ColumnPicker
            title="Output columns (one per graded question)"
            columns={columns}
            selected={outputCols}
            onToggle={toggle(outputCols, setOutputCols)}
          />
          <div>
            <h3 className="mb-1 font-medium">Course (optional)</h3>
            <input
              className="w-full rounded border p-2"
              placeholder="e.g. CS101"
              value={course}
              onChange={(e) => setCourse(e.target.value)}
            />
          </div>
          <div>
            <h3 className="mb-1 font-medium">Exam name (optional)</h3>
            <input
              className="w-full rounded border p-2"
              placeholder="auto-generated if blank"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <button
            onClick={create}
            className="rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700"
          >
            Create exam
          </button>
        </>
      )}
    </div>
  )
}
