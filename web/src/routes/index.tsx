import { Plus, Archive, FileText } from 'lucide-react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useCallback, useEffect, useState } from 'react'
import { api, type ExamSummary, type WorkspaceInfo } from '#/lib/api'
import { Button } from '#/components/ui/button'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '#/components/ui/card'
import { Alert, AlertDescription } from '#/components/ui/alert'
import { Skeleton } from '#/components/ui/skeleton'

export const Route = createFileRoute('/')({ component: Home })

interface LegacyListProps {
  legacy: WorkspaceInfo[]
  legacyCount: number
  onImportWorkspace: (name: string) => void
  onImportLegacySessions: () => void
}

function LegacyList({ legacy, legacyCount, onImportWorkspace, onImportLegacySessions }: LegacyListProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Archive className="h-5 w-5" />
          Legacy data
        </CardTitle>
        <CardDescription>
          Import grading sessions from the old Python app format
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {legacyCount > 0 && (
          <Button onClick={onImportLegacySessions} variant="secondary" size="sm">
            Import {legacyCount} loose session(s)
          </Button>
        )}
        {legacy.map((w) => (
          <div key={w.name} className="flex items-center justify-between">
            <span className="text-sm">{w.name} ({w.exams} exam(s))</span>
            <Button onClick={() => onImportWorkspace(w.name)} variant="outline" size="sm">
              Import
            </Button>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

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
      api.legacySessionsCount().catch(() => 0),
    ])
      .then(([e, w, ls]) => {
        setExams(e)
        setLegacy(w)
        setLegacyCount(ls as number)
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => refresh(), [refresh])

  async function importWorkspace(name: string) {
    setError(null)
    try {
      const r = await api.importLegacyWorkspace(name)
      setInfo(`Imported ${r.imported_exams.length} exam(s) from workspace "${name}"`)
      refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function importLegacySessions() {
    setError(null)
    try {
      const r = await api.importLegacySessions()
      setInfo(`Imported ${r.imported_exams.length} exam(s) from loose sessions`)
      refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const hasLegacy = legacy.length > 0 || legacyCount > 0

  return (
    <div className="mx-auto max-w-3xl p-8">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Tentanator</h1>
        <Link to="/new">
          <Button>
            <Plus className="mr-1 h-4 w-4" />
            New exam
          </Button>
        </Link>
      </div>

      <h2 className="mt-8 mb-3 text-xl font-semibold">Exams</h2>

      {loading && (
        <div className="space-y-3">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-3/4" />
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertDescription>
            {error}. Is the backend running?
          </AlertDescription>
        </Alert>
      )}

      {info && (
        <Alert>
          <AlertDescription>{info}</AlertDescription>
        </Alert>
      )}

      {!loading && !error && exams.length === 0 && (
        <p className="text-muted-foreground">No exams yet. Create one to start grading.</p>
      )}

      {exams.length > 0 && (
        <div className="space-y-2">
          {exams.map((e) => (
            <Link key={e.name} to="/exam/$name" params={{ name: e.name }}>
              <Card className="transition-colors hover:bg-accent/50">
                <CardContent className="flex items-center justify-between p-4">
                  <div>
                    <div className="font-medium">{e.name}</div>
                    <div className="text-sm text-muted-foreground">
                      {e.exam_file}
                      {e.course ? ` · ${e.course}` : ''}
                      {e.archived ? ' · archived' : ''}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <FileText className="h-4 w-4" />
                    {e.graded_count} graded
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}

      {hasLegacy && (
        <div className="mt-10">
          <LegacyList
            legacy={legacy}
            legacyCount={legacyCount}
            onImportWorkspace={importWorkspace}
            onImportLegacySessions={importLegacySessions}
          />
        </div>
      )}
    </div>
  )
}
