import { Link } from '@tanstack/react-router'
import { Button } from '#/components/ui/button'

export type Tab = 'grade' | 'import' | 'scheme' | 'results' | 'pdf'

const tabs: { id: Tab; label: string; to: string }[] = [
  { id: 'grade', label: 'Grade', to: '/exam/$name' },
  { id: 'import', label: 'Import', to: '/exam/$name/import' },
  { id: 'scheme', label: 'Scheme', to: '/exam/$name/scheme' },
  { id: 'results', label: 'Results', to: '/exam/$name/results' },
  { id: 'pdf', label: 'PDF', to: '/exam/$name/pdf' },
]

export function ExamNav({ name, active }: { name: string; active: Tab }) {
  return (
    <div className="flex items-center justify-between border-b pb-2">
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
      <Link to="/">
        <Button variant="link" size="sm">← exams</Button>
      </Link>
    </div>
  )
}
