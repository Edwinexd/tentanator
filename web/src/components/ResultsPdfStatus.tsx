import { Download, Loader2 } from 'lucide-react'
import type { PdfJob } from '#/lib/api'
import { Alert, AlertDescription } from '#/components/ui/alert'
import { Button } from '#/components/ui/button'

function time(iso: string | null): string {
  return iso ? new Date(iso).toLocaleTimeString() : ''
}

/** Status line for the exam's latest results-PDF job, with a download when done. */
export function ResultsPdfStatus({ job, onDownload }: { job: PdfJob | null; onDownload: () => void }) {
  if (!job) return null
  if (job.status === 'running') {
    return (
      <Alert>
        <AlertDescription className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          Rendering results PDF{job.scanned_pdf ? ` with covers from ${job.scanned_pdf}` : ''} (started{' '}
          {time(job.started_at)}). This runs in the background, so you can leave this page and come back.
        </AlertDescription>
      </Alert>
    )
  }
  if (job.status === 'failed') {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          Results PDF failed ({time(job.finished_at)}): {job.error}
        </AlertDescription>
      </Alert>
    )
  }
  const missing = job.covers_missing.length
  return (
    <Alert>
      <AlertDescription className="flex items-center justify-between gap-2">
        <span>
          PDF ready: {job.filename ?? 'results PDF'} ({job.students ?? '?'} students, finished{' '}
          {time(job.finished_at)})
          {missing > 0 && `. No cover page for ${missing}: ${job.covers_missing.join(', ')}`}
        </span>
        {job.filename && (
          <Button onClick={onDownload} variant="outline" size="sm">
            <Download className="mr-1 h-4 w-4" />
            Download
          </Button>
        )}
      </AlertDescription>
    </Alert>
  )
}
