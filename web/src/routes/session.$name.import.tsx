import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { api, type GradeConflict, type ImportSummary, type Session } from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'

export const Route = createFileRoute('/session/$name/import')({ component: ImportView })

function Badge({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded border px-3 py-1">
      {label}: <span className="font-medium">{value}</span>
    </div>
  )
}

function ImportView() {
  const { name } = Route.useParams()
  const [session, setSession] = useState<Session | null>(null)
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
    api.getSession(name).then(setSession).catch((e: Error) => setError(e.message))
    api.listExams().then(setExams).catch(() => {})
    loadConflicts()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name])

  useEffect(() => {
    setSummary(null)
    setMapping({})
    if (!file) {
      setColumns([])
      return
    }
    api
      .examColumns(file)
      .then((cols) => {
        setColumns(cols)
        setIdColumn(cols[0] ?? '')
      })
      .catch((e: Error) => setError(e.message))
  }, [file])

  function buildReq() {
    const mappings = Object.entries(mapping)
      .filter(([, c]) => c)
      .map(([output_col, column]) => ({ column, output_col }))
    return { file, id_column: idColumn, mappings, label: file }
  }
  async function preview() {
    setError(null)
    setInfo(null)
    try {
      setSummary(await api.importPreview(name, buildReq()))
    } catch (e) {
      setError((e as Error).message)
    }
  }
  async function apply() {
    setError(null)
    try {
      const sum = await api.importApply(name, buildReq())
      setSummary(sum)
      setInfo(`Imported ${sum.new} new, ${sum.same} unchanged, ${sum.conflict} flagged as conflicts`)
      loadConflicts()
    } catch (e) {
      setError((e as Error).message)
    }
  }
  async function resolve(c: GradeConflict, choose: 'existing' | 'incoming') {
    try {
      await api.resolveConflict(name, { output_col: c.output_col, row_id: c.row_id, choose })
      loadConflicts()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const outCols = session?.output_columns ?? []
  return (
    <div className="mx-auto max-w-4xl space-y-5 p-8">
      <ExamNav name={name} active="import" />
      <h1 className="text-2xl font-bold">Import grades</h1>
      {error && <p className="rounded bg-red-100 p-2 text-red-700">{error}</p>}
      {info && <p className="rounded bg-green-100 p-2 text-green-800">{info}</p>}

      <section className="space-y-2">
        <p className="text-sm text-gray-500">
          Pick a graded file (from <code>exams/</code>), map its id column and any grade columns to
          questions, then preview and apply. Conflicting grades are flagged for review.
        </p>
        <div className="flex flex-wrap gap-3">
          <label className="text-sm">
            File{' '}
            <select className="rounded border p-1" value={file} onChange={(e) => setFile(e.target.value)}>
              <option value="">select…</option>
              {exams.map((e) => (
                <option key={e} value={e}>
                  {e}
                </option>
              ))}
            </select>
          </label>
          {columns.length > 0 && (
            <label className="text-sm">
              ID column{' '}
              <select className="rounded border p-1" value={idColumn} onChange={(e) => setIdColumn(e.target.value)}>
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </section>

      {columns.length > 0 && (
        <section>
          <h2 className="mb-2 font-semibold">Column → question mapping</h2>
          <table className="text-sm">
            <tbody>
              {outCols.map((col) => (
                <tr key={col}>
                  <td className="py-1 pr-3">{col}</td>
                  <td>
                    <select
                      className="rounded border p-1"
                      value={mapping[col] ?? ''}
                      onChange={(e) => setMapping((m) => ({ ...m, [col]: e.target.value }))}
                    >
                      <option value="">(skip)</option>
                      {columns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-2 flex gap-2">
            <button onClick={preview} className="rounded border px-3 py-1 text-sm hover:bg-gray-50">
              Preview
            </button>
            <button
              onClick={apply}
              className="rounded bg-blue-600 px-3 py-1 text-sm font-medium text-white hover:bg-blue-700"
            >
              Apply
            </button>
          </div>
        </section>
      )}

      {summary && (
        <section className="text-sm">
          <h2 className="mb-1 font-semibold">Summary</h2>
          <div className="flex flex-wrap gap-2">
            <Badge label="new" value={summary.new} />
            <Badge label="unchanged" value={summary.same} />
            <Badge label="conflicts" value={summary.conflict} />
            <Badge label="skipped" value={summary.skipped} />
            <Badge label="unknown ids" value={summary.unknown_ids} />
          </div>
        </section>
      )}

      {conflicts.length > 0 && (
        <section>
          <h2 className="mb-2 font-semibold">Unresolved conflicts ({conflicts.length})</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-gray-500">
                <th className="py-1">Question</th>
                <th>Student</th>
                <th>Existing</th>
                <th>Incoming</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {conflicts.map((c) => (
                <tr key={`${c.output_col}|${c.row_id}|${c.incoming_source}`} className="border-b">
                  <td className="py-1">{c.output_col}</td>
                  <td>{c.row_id}</td>
                  <td>
                    {c.existing_grade}{' '}
                    <span className="text-xs text-gray-400">({c.existing_source || 'manual'})</span>
                  </td>
                  <td>
                    {c.incoming_grade} <span className="text-xs text-gray-400">({c.incoming_source})</span>
                  </td>
                  <td className="text-right">
                    <button
                      onClick={() => resolve(c, 'existing')}
                      className="mr-1 rounded border px-2 py-0.5 text-xs hover:bg-gray-50"
                    >
                      keep existing
                    </button>
                    <button
                      onClick={() => resolve(c, 'incoming')}
                      className="rounded border px-2 py-0.5 text-xs hover:bg-gray-50"
                    >
                      use incoming
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  )
}
