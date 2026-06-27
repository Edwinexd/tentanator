import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'
import {
  api,
  type GlobalBankMatch,
  type GlobalBankStatus,
} from '#/lib/api'
import { Button } from '#/components/ui/button'
import { Input } from '#/components/ui/input'
import { Label } from '#/components/ui/label'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '#/components/ui/card'
import { Badge } from '#/components/ui/badge'
import { Alert, AlertDescription } from '#/components/ui/alert'
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '#/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '#/components/ui/table'
import { toast } from '#/hooks/use-toast'
import { ArrowLeft, Database, RefreshCw, Upload, Search, Loader2 } from 'lucide-react'

export const Route = createFileRoute('/global-bank')({ component: GlobalBank })

// 'auto' maps to undefined (let the backend pick the language).
type LangChoice = 'auto' | 'se' | 'en'

function questionFor(m: GlobalBankMatch, lang: LangChoice): string {
  if (lang === 'se') return m.q_se
  if (lang === 'en') return m.q_en
  return m.q_en || m.q_se
}

function answerFor(m: GlobalBankMatch, lang: LangChoice): string {
  if (lang === 'se') return m.ans_se
  if (lang === 'en') return m.ans_en
  return m.ans_en || m.ans_se
}

function GlobalBank() {
  const [status, setStatus] = useState<GlobalBankStatus | null>(null)
  const [query, setQuery] = useState('')
  const [lang, setLang] = useState<LangChoice>('auto')
  const [matches, setMatches] = useState<GlobalBankMatch[]>([])
  const [searched, setSearched] = useState(false)
  const [searching, setSearching] = useState(false)
  const [reindexing, setReindexing] = useState(false)
  const [importing, setImporting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  function loadStatus() {
    api
      .globalBankStatus()
      .then(setStatus)
      .catch((e: Error) => setError(e.message))
  }

  useEffect(() => loadStatus(), [])

  async function search() {
    const q = query.trim()
    if (!q) return
    setError(null)
    setSearching(true)
    try {
      const res = await api.globalBankSearch(q, lang === 'auto' ? undefined : lang)
      setMatches(res.matches)
      setSearched(true)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSearching(false)
    }
  }

  async function reindex() {
    setReindexing(true)
    try {
      const res = await api.globalBankReindex()
      toast({
        title: 'Reindexed',
        description: `Embedded ${res.embedded} of ${res.total_questions} question(s).`,
      })
      loadStatus()
    } catch (e) {
      toast({ variant: 'destructive', title: 'Reindex failed', description: (e as Error).message })
    } finally {
      setReindexing(false)
    }
  }

  async function onImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = '' // allow re-selecting the same file
    if (!file) return
    setImporting(true)
    try {
      const up = await api.uploadRawFile(file)
      const res = await api.globalBankImport(up.filename)
      toast({
        title: 'Bank imported',
        description: `Imported ${res.imported} question(s) into '${res.bank}'. Reindex to embed.`,
      })
      loadStatus()
    } catch (e) {
      toast({ variant: 'destructive', title: 'Import failed', description: (e as Error).message })
    } finally {
      setImporting(false)
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div className="flex items-center gap-2">
        <Link to="/">
          <Button variant="ghost" size="icon" aria-label="Back">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <h1 className="flex items-center gap-2 text-2xl font-bold">
          <Database className="h-6 w-6" />
          Global question bank
        </h1>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Status</CardTitle>
          <CardDescription>Shared question bank used for auto-matching across exams</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {status ? (
            <>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">{status.total_questions} questions</Badge>
                <Badge variant="secondary">{status.indexed_vectors} indexed vectors</Badge>
              </div>
              {status.banks.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {status.banks.map((b) => (
                    <Badge key={b.name} variant="outline">
                      {b.name} ({b.questions})
                    </Badge>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No banks loaded yet.</p>
              )}
            </>
          ) : (
            <p className="text-muted-foreground">Loading…</p>
          )}

          <div className="flex flex-wrap gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls"
              className="hidden"
              onChange={onImportFile}
            />
            <Button
              onClick={() => fileInputRef.current?.click()}
              disabled={importing}
              variant="outline"
              size="sm"
            >
              {importing ? (
                <Loader2 className="mr-1 h-4 w-4 animate-spin" />
              ) : (
                <Upload className="mr-1 h-4 w-4" />
              )}
              Import CSV
            </Button>
            <Button onClick={reindex} disabled={reindexing} variant="outline" size="sm">
              {reindexing ? (
                <Loader2 className="mr-1 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-1 h-4 w-4" />
              )}
              Reindex
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Search</CardTitle>
          <CardDescription>Find semantically similar questions in the bank</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-2">
            <div className="flex-1 space-y-2">
              <Label htmlFor="bank-query">Query</Label>
              <Input
                id="bank-query"
                placeholder="e.g. explain version control branching"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') void search() }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="bank-lang">Language</Label>
              <Select value={lang} onValueChange={(v) => setLang(v as LangChoice)}>
                <SelectTrigger id="bank-lang" className="w-36">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto</SelectItem>
                  <SelectItem value="se">Swedish</SelectItem>
                  <SelectItem value="en">English</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button onClick={search} disabled={searching || !query.trim()}>
              {searching ? (
                <Loader2 className="mr-1 h-4 w-4 animate-spin" />
              ) : (
                <Search className="mr-1 h-4 w-4" />
              )}
              Search
            </Button>
          </div>

          {searched && matches.length === 0 && !searching && (
            <p className="text-sm text-muted-foreground">No matches found.</p>
          )}

          {matches.length > 0 && (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Score</TableHead>
                  <TableHead>QID</TableHead>
                  <TableHead>Question</TableHead>
                  <TableHead>Answer</TableHead>
                  <TableHead>Chapter / Subject</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {matches.map((m) => (
                  <TableRow key={`${m.bank}::${m.qid}`}>
                    <TableCell className="font-mono text-xs">{m.score.toFixed(3)}</TableCell>
                    <TableCell className="font-mono text-xs">{m.qid}</TableCell>
                    <TableCell className="max-w-sm whitespace-pre-wrap">
                      {questionFor(m, lang) || '—'}
                    </TableCell>
                    <TableCell className="max-w-sm whitespace-pre-wrap text-muted-foreground">
                      {answerFor(m, lang) || '—'}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {[m.chapter, m.subject].filter(Boolean).join(' / ') || '—'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
