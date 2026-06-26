import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState, type ChangeEvent } from 'react'
import { api } from '#/lib/api'
import { ExamNav } from '#/components/ExamNav'

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
      await api.uploadFile('scans', f)
      setScans(await api.listScans())
      setScan(f.name)
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
      const r = await api.exportResultsPdf(name, withCover ? scan : undefined)
      const miss = r.covers_missing?.length
        ? ` (${r.covers_missing.length} without a detected cover page)`
        : ''
      setPdfPath(r.path)
      setInfo(`Generated ${r.path} — ${r.students} students${miss}`)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function download() {
    if (!pdfPath) return
    const filename = pdfPath.split('/').pop()
    if (!filename) return
    setError(null)
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
      <p className="text-sm text-gray-500">
        One continuous PDF: a LaTeX answer sheet per student (responses, marks, grade, id barcode),
        with the original scanned cover page prepended when a scanned exam PDF is provided.
      </p>

      <section className="space-y-2">
        <h2 className="font-semibold">1. Scanned exam PDF (cover pages)</h2>
        <div className="flex flex-wrap items-center gap-3">
          {scans.length > 0 && (
            <select className="rounded border p-2" value={scan} onChange={(e) => setScan(e.target.value)}>
              <option value="">select a scanned PDF…</option>
              {scans.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          )}
          <label className="text-sm text-gray-600">
            {scans.length > 0 ? 'or upload' : 'upload a scanned exam PDF'}{' '}
            <input type="file" accept="application/pdf,.pdf" onChange={onUpload} className="text-sm" />
          </label>
        </div>
      </section>

      <section className="space-y-2">
        <h2 className="font-semibold">2. Generate</h2>
        <div className="flex gap-2">
          <button
            disabled={busy || !scan}
            onClick={() => generate(true)}
            className="rounded bg-blue-600 px-3 py-1 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            Generate with cover pages
          </button>
          <button
            disabled={busy}
            onClick={() => generate(false)}
            className="rounded border px-3 py-1 text-sm hover:bg-gray-50 disabled:opacity-50"
          >
            Answer sheets only
          </button>
        </div>
      </section>

      {busy && <p className="text-gray-500">Rendering… (this can take a minute)</p>}
      {info && <p className="rounded bg-green-100 p-2 text-green-800">{info}</p>}
      {pdfPath && (
        <button
          onClick={download}
          className="rounded bg-blue-600 px-3 py-1 text-sm font-medium text-white hover:bg-blue-700"
        >
          Download PDF
        </button>
      )}
      {error && <p className="rounded bg-red-100 p-2 text-red-700">{error}</p>}
    </div>
  )
}
