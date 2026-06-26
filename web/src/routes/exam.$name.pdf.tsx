import { FileText, Upload, Download, Printer, Scan } from 'lucide-react'
import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState, type ChangeEvent } from 'react'
import { api } from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'
import { Button } from '#/components/ui/button'
import { Label } from '#/components/ui/label'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from '#/components/ui/card'
import { Alert, AlertDescription } from '#/components/ui/alert'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '#/components/ui/select'

export const Route = createFileRoute('/exam/$name/pdf')({ component: PdfView })

function PdfView() {
  const { name } = Route.useParams()
  const [scans, setScans] = useState<string[]>([])
  const [scan, setScan] = useState('')
  const [busy, setBusy] = useState(false)
  const [info, setInfo] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pdfPath, setPdfPath] = useState<string | null>(null)

  useEffect(() => {
    api.listScans().then(setScans).catch(() => {})
  }, [])

  async function onUpload(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    setError(null)
    try {
      const { filename } = await api.uploadScan(f)
      setScan(filename)
      setScans((prev) => (prev.includes(filename) ? prev : [...prev, filename]))
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function generate(withCover: boolean) {
    setBusy(true)
    setError(null)
    setInfo(null)
    setPdfPath(null)
    try {
      // Cover pages are prepended when a scanned PDF is supplied to the
      // renderer; the "with cover" button opts into passing the scan.
      const result = await api.exportResultsPdf(name, withCover ? scan || undefined : undefined)
      const path = typeof result.path === 'string' ? result.path : null
      if (path) {
        setPdfPath(path)
        setInfo(`PDF generated: ${path}`)
      } else {
        setInfo('PDF generated.')
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function download() {
    if (!pdfPath) return
    // pdfPath is the renderer's absolute server path; the file lives in
    // graded_exams/, served by the backend at /api/graded/{filename}.
    const filename = pdfPath.split('/').pop() ?? 'results.pdf'
    try {
      await api.downloadGraded(filename)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-4 p-8">
      <ExamNav name={name} active="pdf" />
      <h1 className="text-2xl font-bold">Results PDF</h1>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Printer className="h-5 w-5" />
            Generate answer sheets
          </CardTitle>
          <CardDescription>
            Render LaTeX answer sheets with student responses, marks, and grades
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Scanned exam PDF (optional — prepends cover pages matched by barcode)</Label>
            <Select value={scan} onValueChange={setScan}>
              <SelectTrigger>
                <SelectValue placeholder="No scan selected" />
              </SelectTrigger>
              <SelectContent>
                {scans.map((f) => (
                  <SelectItem key={f} value={f}>{f}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Upload className="h-4 w-4" />
            <span>Or upload a scan:</span>
            <input
              type="file"
              accept=".pdf"
              onChange={onUpload}
              className="text-sm file:mr-2 file:rounded file:border-0 file:bg-primary file:px-3 file:py-1 file:text-xs file:text-primary-foreground"
            />
          </div>
        </CardContent>
        <CardFooter className="flex-wrap gap-2">
          <Button disabled={busy} onClick={() => generate(false)}>
            <FileText className="mr-1 h-4 w-4" />
            {busy ? 'Generating…' : 'Generate PDF'}
          </Button>
          <Button
            disabled={busy || !scan}
            onClick={() => generate(true)}
            variant="secondary"
          >
            <Scan className="mr-1 h-4 w-4" />
            {busy ? 'Generating…' : 'With cover pages'}
          </Button>
        </CardFooter>
      </Card>

      {pdfPath && (
        <Alert>
          <AlertDescription className="flex items-center justify-between">
            <span>PDF ready: {pdfPath}</span>
            <Button onClick={download} variant="outline" size="sm">
              <Download className="mr-1 h-4 w-4" />
              Download
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {info && <Alert><AlertDescription>{info}</AlertDescription></Alert>}
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
    </div>
  )
}
