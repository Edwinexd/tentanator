import { Plus, Archive, ArchiveRestore, FileText, Database, Combine } from 'lucide-react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useCallback, useEffect, useState, type MouseEvent } from 'react'
import { api, type ExamSummary, type WorkspaceInfo } from '#/lib/api'
import { Button } from '#/components/ui/button'
import { Checkbox } from '#/components/ui/checkbox'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '#/components/ui/card'
import { Alert, AlertDescription } from '#/components/ui/alert'
import { Skeleton } from '#/components/ui/skeleton'
import { PageShell } from '#/components/PageShell'

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

interface ExamCardProps {
  exam: ExamSummary
  onArchive?: () => void
  onUnarchive?: () => void
}

function ExamCard({ exam: e, onArchive, onUnarchive }: ExamCardProps) {
  const handle = (fn?: () => void) => (ev: MouseEvent) => {
    ev.preventDefault()
    ev.stopPropagation()
    fn?.()
  }
  return (
    <Link to="/exam/$name" params={{ name: e.name }}>
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
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <FileText className="h-4 w-4" />
              {e.graded_count} graded
            </span>
            {onArchive && (
              <Button variant="ghost" size="sm" onClick={handle(onArchive)}>
                <Archive className="mr-1 h-3 w-3" />
                Archive
              </Button>
            )}
            {onUnarchive && (
              <Button variant="ghost" size="sm" onClick={handle(onUnarchive)}>
                <ArchiveRestore className="mr-1 h-3 w-3" />
                Unarchive
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}

function Home() {
  const [exams, setExams] = useState<ExamSummary[]>([])
  const [archived, setArchived] = useState<ExamSummary[]>([])
  const [showArchived, setShowArchived] = useState(false)
  const [legacy, setLegacy] = useState<WorkspaceInfo[]>([])
  const [legacyCount, setLegacyCount] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(() => {
    Promise.all([
      api.listExams(),
      api.listLegacyWorkspaces().catch(() => [] as WorkspaceInfo[]),
      api.legacySessionsInfo().then((r) => r.count).catch(() => 0),
    ])
      .then(([e, w, ls]) => {
        setExams(e)
        setLegacy(w)
        setLegacyCount(ls as number)
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const refreshArchived = useCallback(() => {
    api
      .listExams({ archived: true })
      .then(setArchived)
      .catch((e: Error) => setError(e.message))
  }, [])

  useEffect(() => refresh(), [refresh])

  useEffect(() => {
    if (showArchived) refreshArchived()
  }, [showArchived, refreshArchived])

  async function archiveExam(name: string) {
    setError(null)
    try {
      await api.archiveExam(name)
      refresh()
      if (showArchived) refreshArchived()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function unarchiveExam(name: string) {
    setError(null)
    try {
      await api.unarchiveExam(name)
      refresh()
      refreshArchived()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function importWorkspace(name: string) {
    setError(null)
    try {
      const r = await api.importWorkspace(name)
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
    <PageShell className="space-y-0">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Tentanator</h1>
        <div className="flex items-center gap-2">
          <Link to="/combine">
            <Button variant="outline">
              <Combine className="mr-1 h-4 w-4" />
              Combine Moodle
            </Button>
          </Link>
          <Link to="/global-bank">
            <Button variant="outline">
              <Database className="mr-1 h-4 w-4" />
              Question bank
            </Button>
          </Link>
          <Link to="/new">
            <Button>
              <Plus className="mr-1 h-4 w-4" />
              New exam
            </Button>
          </Link>
        </div>
      </div>

      <div className="mt-8 mb-3 flex items-center justify-between">
        <h2 className="text-xl font-semibold">Exams</h2>
        <label className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <Checkbox
            checked={showArchived}
            onCheckedChange={(c) => setShowArchived(c === true)}
          />
          <span>Show archived</span>
        </label>
      </div>

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
            <ExamCard
              key={e.name}
              exam={e}
              onArchive={() => archiveExam(e.name)}
            />
          ))}
        </div>
      )}

      {showArchived && (
        <div className="mt-8">
          <h2 className="mb-3 text-xl font-semibold text-muted-foreground">Archived</h2>
          {archived.length === 0 ? (
            <p className="text-sm text-muted-foreground">No archived exams.</p>
          ) : (
            <div className="space-y-2">
              {archived.map((e) => (
                <ExamCard
                  key={e.name}
                  exam={e}
                  onUnarchive={() => unarchiveExam(e.name)}
                />
              ))}
            </div>
          )}
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
    </PageShell>
  )
}
