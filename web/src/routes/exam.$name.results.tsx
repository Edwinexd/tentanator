import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { api, type ResultsResponse } from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'

export const Route = createFileRoute('/exam/$name/results')({ component: ResultsView })

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded border px-3 py-1">
      <span className="text-gray-500">{label}:</span>{' '}
      <span className="font-medium">{value}</span>
    </div>
  )
}

function ResultsView() {
  const { name } = Route.useParams()
  const [data, setData] = useState<ResultsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)

  useEffect(() => {
    api.getResults(name).then(setData).catch((e: Error) => setError(e.message))
  }, [name])

  async function doExport(fn: (n: string) => Promise<{ path: string }>, label: string) {
    setError(null)
    try {
      const { path } = await fn(name)
      setInfo(`${label} → ${path}`)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-4 p-8">
      <ExamNav name={name} active="results" />
      <h1 className="text-2xl font-bold">Results</h1>

      <div className="flex flex-wrap gap-2">
        <button onClick={() => doExport(api.exportDaisy, 'Daisy import')} className="rounded border px-3 py-1 text-sm hover:bg-gray-50">
          Export Daisy (id,grade)
        </button>
        <button onClick={() => doExport(api.exportCsv, 'Per-question CSV')} className="rounded border px-3 py-1 text-sm hover:bg-gray-50">
          Export per-question CSV
        </button>
        <button onClick={() => doExport(api.exportExam, 'Graded xlsx')} className="rounded border px-3 py-1 text-sm hover:bg-gray-50">
          Export full graded xlsx
        </button>
      </div>

      {info && <p className="rounded bg-green-100 p-2 text-green-800">{info}</p>}
      {error && <p className="rounded bg-red-100 p-2 text-red-700">{error}</p>}
      {!data && !error && <p className="text-gray-500">Loading…</p>}
      {data && !data.has_scheme && (
        <p className="text-gray-500">
          No grade scheme yet — configure one on the Scheme tab to compute final grades.
        </p>
      )}

      {data && data.has_scheme && (
        <>
          <div className="flex flex-wrap gap-2 text-sm">
            <Stat label="Students" value={data.total_students} />
            <Stat label="Fully graded" value={`${data.complete}/${data.total_students}`} />
            {data.unresolved_conflicts > 0 && (
              <Stat label="Unresolved conflicts" value={data.unresolved_conflicts} />
            )}
            {Object.entries(data.distribution)
              .sort()
              .map(([g, c]) => (
                <Stat key={g} label={g} value={c} />
              ))}
          </div>

          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-gray-500">
                <th className="py-1">ID</th>
                <th>Total</th>
                <th>Grade</th>
                <th>Complete</th>
              </tr>
            </thead>
            <tbody>
              {data.results.map((r) => (
                <tr key={r.id} className="border-b">
                  <td className="py-1">{r.id}</td>
                  <td>{r.total.toFixed(2)}</td>
                  <td className="font-medium">
                    {r.grade}
                    {r.estimated.length > 0 && (
                      <span className="ml-1 text-xs text-amber-600">(est)</span>
                    )}
                  </td>
                  <td>{r.complete ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}
