import { createFileRoute, Link } from '@tanstack/react-router'
import { useCallback, useEffect, useState } from 'react'
import { api, type SessionSummary, type WorkspaceInfo } from '#/lib/api'

export const Route = createFileRoute('/')({ component: Home })

function Home() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [legacy, setLegacy] = useState<WorkspaceInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(() => {
    Promise.all([api.listSessions(), api.listLegacyWorkspaces().catch(() => [])])
      .then(([s, w]) => {
        setSessions(s)
        setLegacy(w)
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
        `Imported ${r.imported_sessions.length} session(s) and ${r.imported_exams} exam(s) from ${name}` +
          (r.skipped_exams ? ` (${r.skipped_exams} exam(s) already present)` : ''),
      )
      refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="mx-auto max-w-3xl p-8">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Tentanator</h1>
        <Link
          to="/new"
          className="rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700"
        >
          New session
        </Link>
      </div>

      <h2 className="mt-8 mb-3 text-xl font-semibold">Sessions</h2>

      {loading && <p className="text-gray-500">Loading…</p>}
      {error && (
        <p className="rounded bg-red-100 p-3 text-red-700">
          {error}. Is the backend running?
        </p>
      )}
      {info && <p className="rounded bg-green-100 p-3 text-green-800">{info}</p>}
      {!loading && !error && sessions.length === 0 && (
        <p className="text-gray-500">No sessions yet. Create one to start grading.</p>
      )}

      <ul className="divide-y rounded border">
        {sessions.map((s) => (
          <li key={s.session_name}>
            <Link
              to="/session/$name"
              params={{ name: s.session_name }}
              className="flex items-center justify-between px-4 py-3 hover:bg-gray-50"
            >
              <div>
                <div className="font-medium">{s.session_name}</div>
                <div className="text-sm text-gray-500">
                  {s.csv_file} · {s.num_questions} question(s) · updated{' '}
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

      {legacy.length > 0 && (
        <div className="mt-10">
          <h2 className="mb-1 text-xl font-semibold">Import legacy workspaces</h2>
          <p className="mb-3 text-sm text-gray-500">
            Old <code>workspaces/&lt;name&gt;/</code> folders. Importing copies their
            sessions and exams into the store and tags the sessions with the
            workspace name as their course.
          </p>
          <ul className="divide-y rounded border">
            {legacy.map((w) => (
              <li key={w.name} className="flex items-center justify-between px-4 py-3">
                <span>
                  {w.name}{' '}
                  <span className="text-sm text-gray-500">({w.sessions} session(s))</span>
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
        </div>
      )}
    </div>
  )
}
