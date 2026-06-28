import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useState } from 'react'
import { api, type CombineMoodleResp } from '#/lib/api'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from '#/components/ui/card'
import { Button } from '#/components/ui/button'
import { Input } from '#/components/ui/input'
import { Label } from '#/components/ui/label'
import { Badge } from '#/components/ui/badge'
import { Alert, AlertDescription } from '#/components/ui/alert'
import { PageShell, PageHeader } from '#/components/PageShell'
import { Combine, Loader2 } from 'lucide-react'

export const Route = createFileRoute('/combine')({ component: CombineMoodle })

function CombineMoodle() {
  const navigate = useNavigate()
  const [gradesFile, setGradesFile] = useState<File | null>(null)
  const [responsesFile, setResponsesFile] = useState<File | null>(null)
  const [outputName, setOutputName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<CombineMoodleResp | null>(null)

  async function combine() {
    if (!gradesFile) return setError('Choose a grades export file')
    if (!responsesFile) return setError('Choose a responses export file')
    setError(null)
    setResult(null)
    setBusy(true)
    try {
      const grades = await api.uploadRawFile(gradesFile)
      const responses = await api.uploadRawFile(responsesFile)
      const res = await api.combineMoodle(
        grades.filename,
        responses.filename,
        outputName.trim() || undefined,
      )
      setResult(res)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <PageShell>
      <PageHeader title="Combine Moodle dumps" icon={Combine} />

      <Card>
        <CardHeader>
          <CardTitle>Raw Moodle exports</CardTitle>
          <CardDescription>
            Merge a Moodle grades export and a responses export into one exam file ready for grading.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="grades-file">Grades file</Label>
            <Input
              id="grades-file"
              type="file"
              accept=".xlsx,.csv"
              onChange={(e) => setGradesFile(e.target.files?.[0] ?? null)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="responses-file">Responses file</Label>
            <Input
              id="responses-file"
              type="file"
              accept=".xlsx,.csv"
              onChange={(e) => setResponsesFile(e.target.files?.[0] ?? null)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="output-name">Output name (optional)</Label>
            <Input
              id="output-name"
              placeholder="e.g. midterm-2026-combined"
              value={outputName}
              onChange={(e) => setOutputName(e.target.value)}
            />
          </div>
        </CardContent>
        <CardFooter className="flex-col items-stretch gap-3">
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          <div className="flex justify-end">
            <Button onClick={combine} disabled={busy}>
              {busy ? (
                <Loader2 className="mr-1 h-4 w-4 animate-spin" />
              ) : (
                <Combine className="mr-1 h-4 w-4" />
              )}
              Combine
            </Button>
          </div>
        </CardFooter>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Combined</CardTitle>
            <CardDescription>
              Saved as <span className="font-mono">{result.filename}</span> — it now appears as a
              selectable file when creating an exam.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary">{result.students} students</Badge>
              <Badge variant="secondary">{result.questions} questions</Badge>
              {result.dropped_columns.length > 0 && (
                <Badge variant="outline">{result.dropped_columns.length} dropped column(s)</Badge>
              )}
            </div>
            {result.dropped_columns.length > 0 && (
              <div className="space-y-1">
                <div className="text-sm text-muted-foreground">Dropped columns:</div>
                <div className="flex flex-wrap gap-1">
                  {result.dropped_columns.map((c) => (
                    <Badge key={c} variant="outline" className="font-mono text-xs">
                      {c}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            <div className="flex justify-end">
              <Button onClick={() => navigate({ to: '/new' })}>
                Create exam from this file
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </PageShell>
  )
}
