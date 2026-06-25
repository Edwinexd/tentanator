import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { api, type SessionSummary } from '#/lib/api'

export const Route = createFileRoute('/')({ component: Home })

function Home() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api
      .listSessions()
      .then(setSessions)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

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
      {!loading && !error && sessions.length === 0 && (
        <p className="text-gray-500">No sessions yet. Create one to start grading.</p>
      )}

      <ul className="divide-y rounded border">
        {sessions.map((s) => (
          <li key={s.session_name}>
            <Link
              to="/session/$name"
              params={{ name: s.session_name }}
              className="block px-4 py-3 hover:bg-gray-50"
            >
              <div className="font-medium">{s.session_name}</div>
              <div className="text-sm text-gray-500">
                {s.csv_file} · {s.num_questions} question(s) · updated{' '}
                {s.last_updated.slice(0, 19)}
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
