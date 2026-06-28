import { createFileRoute } from '@tanstack/react-router'
import { Fragment, useEffect, useMemo, useState } from 'react'
import { api, type RenderData, type RenderQuestion } from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'
import { PageShell } from '#/components/PageShell'
import { Badge } from '#/components/ui/badge'
import { Alert, AlertDescription } from '#/components/ui/alert'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '#/components/ui/table'
import { ChevronDown, ChevronRight } from 'lucide-react'

export const Route = createFileRoute('/exam/$name/table')({ component: TableView })

function cellPoints(q: RenderQuestion): string {
  if (q.points == null) return '-'
  return `${q.points}/${q.max}`
}

function TableView() {
  const { name } = Route.useParams()
  const [data, setData] = useState<RenderData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<Record<string, boolean>>({})

  useEffect(() => {
    let active = true
    api
      .renderData(name)
      .then((r) => { if (active) setData(r) })
      .catch((e: Error) => { if (active) setError(e.message) })
    return () => { active = false }
  }, [name])

  // All students share the same question order; take the headers from the first.
  const headers = useMemo(() => data?.students[0]?.questions ?? [], [data])

  function toggle(id: string) {
    setOpen((o) => ({ ...o, [id]: !o[id] }))
  }

  // chevron + ID + Grade + Total + one per question
  const colCount = 4 + headers.length

  return (
    <PageShell>
      <ExamNav name={name} active="table" />
      <h1 className="text-2xl font-bold">Table</h1>
      <p className="text-sm text-muted-foreground">
        Every student and the points awarded per question (including auto-zeroed
        columns). Click a row to read each response in full.
      </p>

      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      {!data && !error && <p className="text-muted-foreground">Loading…</p>}

      {data && data.students.length === 0 && (
        <p className="text-muted-foreground">No students in this exam yet.</p>
      )}

      {data && data.students.length > 0 && (
        <>
          <Badge variant="secondary">{data.students.length} students</Badge>
          <div className="overflow-x-auto rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8" />
                  <TableHead>ID</TableHead>
                  <TableHead>Grade</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                  {headers.map((q, i) => (
                    <TableHead key={i} className="whitespace-nowrap text-right" title={q.label}>
                      {q.label || `Q${i + 1}`}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.students.map((s) => {
                  const expanded = open[s.id] ?? false
                  return (
                    <Fragment key={s.id}>
                      <TableRow
                        className="cursor-pointer"
                        onClick={() => toggle(s.id)}
                      >
                        <TableCell className="text-muted-foreground">
                          {expanded ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                        </TableCell>
                        <TableCell className="font-mono text-xs">{s.id}</TableCell>
                        <TableCell>{s.grade || '-'}</TableCell>
                        <TableCell className="text-right">{s.total.toFixed(1)}</TableCell>
                        {s.questions.map((q, i) => (
                          <TableCell
                            key={i}
                            className={
                              'text-right tabular-nums' +
                              (q.estimated ? ' italic text-muted-foreground' : '')
                            }
                            title={q.estimated ? 'estimated (ungraded)' : undefined}
                          >
                            {cellPoints(q)}
                          </TableCell>
                        ))}
                      </TableRow>
                      {expanded && (
                        <TableRow>
                          <TableCell colSpan={colCount} className="bg-muted/40 p-0">
                            <div className="space-y-3 p-4">
                              {s.questions.map((q, i) => (
                                <div key={i} className="space-y-1">
                                  <div className="flex items-center gap-2 text-sm font-medium">
                                    <span>{q.label || `Q${i + 1}`}</span>
                                    <Badge variant="outline">{cellPoints(q)}</Badge>
                                    {q.estimated && <Badge variant="secondary">estimated</Badge>}
                                    {q.group && (
                                      <span className="text-xs text-muted-foreground">{q.group}</span>
                                    )}
                                  </div>
                                  <div className="whitespace-pre-wrap rounded-md border bg-background p-2 text-sm">
                                    {q.response || (
                                      <span className="text-muted-foreground">(no response)</span>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </TableCell>
                        </TableRow>
                      )}
                    </Fragment>
                  )
                })}
              </TableBody>
            </Table>
          </div>
        </>
      )}
    </PageShell>
  )
}
