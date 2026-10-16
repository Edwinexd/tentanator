import { useCallback, useEffect, useState } from 'react'
import { api, type PdfJob } from '#/lib/api'

const POLL_MS = 3000

/**
 * The exam's results-PDF job. The render runs in the backend, so the page can
 * be left and revisited: on mount this picks up the latest job and polls while
 * it is running.
 */
export function useResultsPdfJob(name: string) {
  const [job, setJob] = useState<PdfJob | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    api
      .resultsPdfJob(name)
      .then((j) => { if (active) setJob(j) })
      .catch(() => {})
    return () => { active = false }
  }, [name])

  const running = job?.status === 'running'
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => {
      api
        .resultsPdfJob(name)
        .then(setJob)
        .catch((e: Error) => setError(e.message))
    }, POLL_MS)
    return () => clearInterval(id)
  }, [name, running])

  const start = useCallback(
    async (scannedPdf?: string) => {
      setError(null)
      try {
        setJob(await api.exportResultsPdf(name, scannedPdf))
      } catch (e) {
        setError((e as Error).message)
      }
    },
    [name],
  )

  const download = useCallback(async () => {
    if (!job?.filename) return
    try {
      await api.downloadGraded(job.filename)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [job?.filename])

  return { job, running, error, start, download }
}
