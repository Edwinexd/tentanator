import { Link } from '@tanstack/react-router'
import { Button } from '#/components/ui/button'

export type Tab = 'grade' | 'import' | 'scheme' | 'results' | 'table' | 'pdf'

const tabs: { id: Tab; label: string; to: string }[] = [
  { id: 'grade', label: 'Grade', to: '/exam/$name' },
  { id: 'import', label: 'Import', to: '/exam/$name/import' },
  { id: 'scheme', label: 'Scheme', to: '/exam/$name/scheme' },
  { id: 'results', label: 'Results', to: '/exam/$name/results' },
  { id: 'table', label: 'Table', to: '/exam/$name/table' },
  { id: 'pdf', label: 'PDF', to: '/exam/$name/pdf' },
]

export function ExamNav({ name, active }: { name: string; active: Tab }) {
  return (
    <div className="space-y-2 border-b pb-2">
      <div className="flex items-center justify-between gap-2">
        <div className="truncate text-sm text-muted-foreground">
          Exam: <span className="font-semibold text-foreground">{name}</span>
        </div>
        <Link to="/">
          <Button variant="link" size="sm">← exams</Button>
        </Link>
      </div>
      <div className="flex gap-1">
        {tabs.map((t) => (
          <Link key={t.id} to={t.to} params={{ name }}>
            <Button
              variant={active === t.id ? 'default' : 'ghost'}
              size="sm"
            >
              {t.label}
            </Button>
          </Link>
        ))}
      </div>
    </div>
  )
}
