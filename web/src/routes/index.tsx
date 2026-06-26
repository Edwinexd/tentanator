import { createFileRoute, Link } from '@tanstack/react-router'
import { useCallback, useEffect, useState } from 'react'
import { api, type ExamSummary, type WorkspaceInfo } from '#/lib/api'

export const Route = createFileRoute('/')({ component: Home })

function Home() {
  const [exams, setExams] = useState<ExamSummary[]>([])
  const [legacy, setLegacy] = useState<WorkspaceInfo[]>([])
  const [legacyCount, setLegacyCount] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(() => {
    Promise.all([
      api.listExams(),
      api.listLegacyWorkspaces().catch(() => [] as WorkspaceInfo[]),
      api.legacySessionsInfo().catch(() => ({ count: 0 })),
    ])
      .then(([e, w, ls]) => {
        setExams(e)
        setLegacy(w)
        setLegacyCount(ls.count)
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => refresh(), [refresh])

  async function importWorkspace(name: string) {
    setError(null)
    try {
      const r = await api.importWorkspace(name)
      setInfo(
        `Imported ${r.imported_exams.length} exam(s) and ${r.imported_files} file(s) from ${name}` +
          (r.skipped_files ? ` (${r.skipped_files} file(s) already present)` : ''),
      )
      refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function importLegacySessions() {
    setError(null)
    try {
      const r = await api.importLegacySessions()
      setInfo(`Imported ${r.imported_exams.length} legacy exam(s) from .tentanator_sessions/`)
      refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const hasLegacy = legacy.length > 0 || legacyCount > 0

  return (
    <div className="mx-auto max-w-3xl p-8">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Tentanator</h1>
        <Link
          to="/new"
          className="rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700"
        >
          New exam
        </Link>
      </div>

      <h2 className="mt-8 mb-3 text-xl font-semibold">Exams</h2>

      {loading && <p className="text-gray-500">Loading…</p>}
      {error && (
        <p className="rounded bg-red-100 p-3 text-red-700">
          {error}. Is the backend running?
        </p>
      )}
      {info && <p className="rounded bg-green-100 p-3 text-green-800">{info}</p>}
      {!loading && !error && exams.length === 0 && (
        <p className="text-gray-500">No exams yet. Create one to start grading.</p>
      )}

      <ul className="divide-y rounded border">
        {exams.map((s) => (
          <li key={s.name}>
            <Link
              to="/exam/$name"
              params={{ name: s.name }}
              className="flex items-center justify-between px-4 py-3 hover:bg-gray-50"
            >
              <div>
                <div className="font-medium">{s.name}</div>
                <div className="text-sm text-gray-500">
                  {s.exam_file} · {s.num_questions} question(s) · updated{' '}
                  {s.last_updated.slice(0, 19)}
                </div>
              </div>
              {s.course && (
                <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700">
                  {s.course}
                </span>
              )}
            </Link>
          </li>
        ))}
      </ul>

      {hasLegacy && (
        <div className="mt-10">
          <h2 className="mb-1 text-xl font-semibold">Import legacy data</h2>
          <p className="mb-3 text-sm text-gray-500">
            Bring exams from the old layout into the new format. Existing exams and
            files are never overwritten.
          </p>

          {legacyCount > 0 && (
            <div className="mb-3 flex items-center justify-between rounded border px-4 py-3">
              <span>
                <code>.tentanator_sessions/</code>{' '}
                <span className="text-sm text-gray-500">({legacyCount} session(s))</span>
              </span>
              <button
                onClick={importLegacySessions}
                className="rounded border px-3 py-1 text-sm hover:bg-gray-50"
              >
                Import
              </button>
            </div>
          )}

          {legacy.length > 0 && (
            <ul className="divide-y rounded border">
              {legacy.map((w) => (
                <li key={w.name} className="flex items-center justify-between px-4 py-3">
                  <span>
                    {w.name}{' '}
                    <span className="text-sm text-gray-500">({w.exams} exam(s))</span>
                  </span>
                  <button
                    onClick={() => importWorkspace(w.name)}
                    className="rounded border px-3 py-1 text-sm hover:bg-gray-50"
                  >
                    Import
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
