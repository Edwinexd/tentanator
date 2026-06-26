import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import {
  api,
  type Exam,
  type GradeScheme,
  type QuestionConfigUpdate,
  type ResultsResponse,
} from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'

export const Route = createFileRoute('/exam/$name/scheme')({ component: SchemeView })

function defaultScheme(cfg: QuestionConfigUpdate[]): GradeScheme {
  const sum = cfg.map((c) => c.var).filter(Boolean).join(' + ') || '0'
  return {
    constants: [],
    vars: [{ name: 'total', expr: sum }],
    rules: [{ when: 'total >= 0', grade: 'PASS' }],
    total_var: 'total',
    default_grade: 'FAIL',
  }
}

function SchemeView() {
  const { name } = Route.useParams()
  const [cfg, setCfg] = useState<QuestionConfigUpdate[]>([])
  const [schemeText, setSchemeText] = useState('')
  const [preview, setPreview] = useState<ResultsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)

  useEffect(() => {
    api
      .getExam(name)
      .then((s: Exam) => {
        const initial = s.output_columns.map((col, i) => {
          const q = s.questions[col]
          return {
            col,
            var: q?.var || `q${i + 1}`,
            group: q?.group || '',
            qtype: q?.qtype || '',
            max_points: q?.max_points || 0,
            position: q?.position ?? i,
            estimate: q?.estimate || '',
          }
        })
        setCfg(initial)
        setSchemeText(JSON.stringify(s.scheme ?? defaultScheme(initial), null, 2))
      })
      .catch((e: Error) => setError(e.message))
  }, [name])

  function setRow(i: number, patch: Partial<QuestionConfigUpdate>) {
    setCfg((rows) => rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  }
  function parseScheme(): GradeScheme | null {
    try {
      return JSON.parse(schemeText) as GradeScheme
    } catch {
      setError('Scheme JSON is invalid')
      return null
    }
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
    const sc = parseScheme()
    if (!sc) return
    try {
      setPreview(await api.previewResults(name, sc))
    } catch (e) {
      setError((e as Error).message)
    }
  }
  async function saveScheme() {
    setError(null)
    const sc = parseScheme()
    if (!sc) return
    try {
      await api.putScheme(name, sc)
      setInfo('Scheme saved')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-5 p-8">
      <ExamNav name={name} active="scheme" />
      <h1 className="text-2xl font-bold">Grade scheme</h1>
      {error && <p className="rounded bg-red-100 p-2 text-red-700">{error}</p>}
      {info && <p className="rounded bg-green-100 p-2 text-green-800">{info}</p>}

      <section>
        <h2 className="mb-2 font-semibold">Question config</h2>
        <p className="mb-2 text-sm text-gray-500">
          <code>var</code> is the name used in scheme expressions; <code>group</code> is an
          optional section tag (aggregate with <code>groupsum("tag")</code>).
        </p>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-gray-500">
              <th className="py-1">Question</th>
              <th>var</th>
              <th>group</th>
              <th>type</th>
              <th>max</th>
              <th>estimate</th>
            </tr>
          </thead>
          <tbody>
            {cfg.map((r, i) => (
              <tr key={r.col} className="border-b">
                <td className="py-1 pr-2">{r.col}</td>
                <td><input className="w-20 rounded border px-1" value={r.var} onChange={(e) => setRow(i, { var: e.target.value })} /></td>
                <td><input className="w-20 rounded border px-1" value={r.group} onChange={(e) => setRow(i, { group: e.target.value })} /></td>
                <td><input className="w-20 rounded border px-1" value={r.qtype} onChange={(e) => setRow(i, { qtype: e.target.value })} /></td>
                <td><input className="w-16 rounded border px-1" type="number" value={r.max_points} onChange={(e) => setRow(i, { max_points: Number(e.target.value) })} /></td>
                <td><input className="w-40 rounded border px-1" value={r.estimate ?? ''} onChange={(e) => setRow(i, { estimate: e.target.value })} placeholder="optional expr" /></td>
              </tr>
            ))}
          </tbody>
        </table>
        <button onClick={saveConfig} className="mt-2 rounded border px-3 py-1 text-sm hover:bg-gray-50">
          Save question config
        </button>
      </section>

      <section>
        <h2 className="mb-2 font-semibold">Scheme</h2>
        <p className="mb-2 text-sm text-gray-500">
          Named <code>vars</code> (expressions), tunable <code>constants</code>, and ordered{' '}
          <code>rules</code> (<code>when</code> → <code>grade</code>, first match wins).
        </p>
        <textarea
          className="h-72 w-full rounded border p-2 font-mono text-xs"
          value={schemeText}
          onChange={(e) => setSchemeText(e.target.value)}
          spellCheck={false}
        />
        <div className="mt-2 flex gap-2">
          <button onClick={doPreview} className="rounded border px-3 py-1 text-sm hover:bg-gray-50">
            Preview
          </button>
          <button onClick={saveScheme} className="rounded bg-blue-600 px-3 py-1 text-sm font-medium text-white hover:bg-blue-700">
            Save scheme
          </button>
        </div>
      </section>

      {preview && (
        <section>
          <h2 className="mb-2 font-semibold">Live preview</h2>
          <div className="flex flex-wrap gap-2 text-sm">
            <div className="rounded border px-3 py-1">
              Fully graded: <span className="font-medium">{preview.complete}/{preview.total_students}</span>
            </div>
            {Object.entries(preview.distribution)
              .sort()
              .map(([g, c]) => (
                <div key={g} className="rounded border px-3 py-1">
                  {g}: <span className="font-medium">{c}</span>
                </div>
              ))}
          </div>
        </section>
      )}
    </div>
  )
}
