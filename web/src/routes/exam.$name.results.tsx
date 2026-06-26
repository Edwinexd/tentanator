import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { api, type ResultsResponse } from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'
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
    api.getResults(name).then(setData).catch((e: Error) => setError(e.message))
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

  const d = data?.distribution
  return (
    <div className="mx-auto max-w-4xl space-y-4 p-8">
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
        <Button onClick={() => doExport(api.exportResultsPdf)} variant="outline" size="sm">
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
            <Badge variant="secondary">{data.students.length} students</Badge>
            <Badge variant="secondary">{data.conflicts ?? 0} unresolved conflict(s)</Badge>
            {d && (
              <>
                {d.mean != null && <Badge variant="secondary">mean {d.mean.toFixed(1)}</Badge>}
                {d.median != null && <Badge variant="secondary">median {d.median.toFixed(1)}</Badge>}
                {d.min != null && <Badge variant="secondary">min {d.min.toFixed(1)}</Badge>}
                {d.max != null && <Badge variant="secondary">max {d.max.toFixed(1)}</Badge>}
                {d.stdev != null && <Badge variant="secondary">σ {d.stdev.toFixed(1)}</Badge>}
              </>
            )}
          </div>

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
                  {data.students.map((s) => (
                    <TableRow key={s.id}>
                      <TableCell className="font-mono text-xs">{s.id}</TableCell>
                      <TableCell>{s.grade ?? '—'}</TableCell>
                      <TableCell>{s.total?.toFixed(1) ?? '—'}</TableCell>
                      <TableCell>{s.estimate?.toFixed(1) ?? '—'}</TableCell>
                      <TableCell>{s.complete ? '✓' : '…'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
