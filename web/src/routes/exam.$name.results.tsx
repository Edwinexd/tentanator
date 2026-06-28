import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { api, type ResultsResponse } from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'
import { PageShell } from '#/components/PageShell'
import { Button } from '#/components/ui/button'
import { Badge } from '#/components/ui/badge'
import { Alert, AlertDescription } from '#/components/ui/alert'
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from '#/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '#/components/ui/table'
import { Download, DownloadCloud, FileText, TableIcon } from 'lucide-react'

export const Route = createFileRoute('/exam/$name/results')({ component: ResultsView })

function ResultsView() {
  const { name } = Route.useParams()
  const [data, setData] = useState<ResultsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    api
      .getResults(name)
      .then((r) => { if (active) setData(r) })
      .catch((e: Error) => { if (active) setError(e.message) })
    return () => { active = false }
  }, [name])

  async function doExport(fn: (n: string) => Promise<void>) {
    setError(null)
    try {
      await fn(name)
      setInfo('Download started')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function exportPdf() {
    setError(null)
    setInfo(null)
    try {
      const result = await api.exportResultsPdf(name)
      // export/results-pdf returns opaque JSON proxied from the renderer; if it
      // points at a generated file, download it via the graded-files endpoint.
      const ref =
        typeof result.path === 'string'
          ? result.path
          : typeof result.filename === 'string'
            ? result.filename
            : null
      if (ref) {
        const filename = ref.split('/').pop() ?? ref
        await api.downloadGraded(filename)
        setInfo('Download started')
      } else {
        setInfo('Results PDF generated')
      }
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const stats = data?.stats
  return (
    <PageShell>
      <ExamNav name={name} active="results" />
      <h1 className="text-2xl font-bold">Results</h1>

      <div className="flex flex-wrap gap-2">
        <Button onClick={() => doExport(api.exportExam)} variant="outline" size="sm">
          <Download className="mr-1 h-4 w-4" />
          Export XLSX
        </Button>
        <Button onClick={() => doExport(api.exportDaisy)} variant="outline" size="sm">
          <DownloadCloud className="mr-1 h-4 w-4" />
          Export Daisy
        </Button>
        <Button onClick={() => doExport(api.exportCsv)} variant="outline" size="sm">
          <TableIcon className="mr-1 h-4 w-4" />
          Export CSV
        </Button>
        <Button onClick={exportPdf} variant="outline" size="sm">
          <FileText className="mr-1 h-4 w-4" />
          Export results PDF
        </Button>
      </div>

      {info && <Alert><AlertDescription>{info}</AlertDescription></Alert>}
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}

      {!data && !error && <p className="text-muted-foreground">Loading…</p>}

      {data && !data.has_scheme && (
        <p className="text-muted-foreground">
          No grade scheme yet — configure one on the Scheme tab to compute final grades.
        </p>
      )}

      {data && data.has_scheme && (
        <>
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">{data.results.length} students</Badge>
            <Badge variant="secondary">{data.unresolved_conflicts} unresolved conflict(s)</Badge>
            {stats != null && (
              <>
                <Badge variant="secondary">mean {stats.mean.toFixed(1)}</Badge>
                <Badge variant="secondary">median {stats.median.toFixed(1)}</Badge>
                <Badge variant="secondary">min {stats.min.toFixed(1)}</Badge>
                <Badge variant="secondary">max {stats.max.toFixed(1)}</Badge>
                <Badge variant="secondary">σ {stats.stdev.toFixed(1)}</Badge>
              </>
            )}
          </div>

          {Object.keys(data.distribution).length > 0 && (
            <div className="flex flex-wrap gap-2">
              {Object.entries(data.distribution)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([grade, count]) => (
                  <Badge key={grade} variant="outline">
                    {grade}: {count}
                  </Badge>
                ))}
            </div>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Student results</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>ID</TableHead>
                    <TableHead>Grade</TableHead>
                    <TableHead>Total</TableHead>
                    <TableHead>Estimate</TableHead>
                    <TableHead>Complete</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.results.map((s) => (
                    <TableRow key={s.id}>
                      <TableCell className="font-mono text-xs">{s.id}</TableCell>
                      <TableCell>{s.grade || '—'}</TableCell>
                      <TableCell>{s.total.toFixed(1)}</TableCell>
                      <TableCell>
                        {s.estimated.length > 0 ? s.estimated.join(', ') : '—'}
                      </TableCell>
                      <TableCell>{s.complete ? '✓' : '…'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </PageShell>
  )
}
