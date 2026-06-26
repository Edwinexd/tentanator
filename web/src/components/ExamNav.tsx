import { Link } from '@tanstack/react-router'

type Tab = 'grade' | 'import' | 'scheme' | 'results' | 'pdf'

export function ExamNav({ name, active }: { name: string; active: Tab }) {
  const tabs: { id: Tab; label: string; to: string }[] = [
    { id: 'grade', label: 'Grade', to: '/exam/$name' },
    { id: 'import', label: 'Import', to: '/exam/$name/import' },
    { id: 'scheme', label: 'Scheme', to: '/exam/$name/scheme' },
    { id: 'results', label: 'Results', to: '/exam/$name/results' },
    { id: 'pdf', label: 'PDF', to: '/exam/$name/pdf' },
  ]
  return (
    <div className="flex items-center justify-between border-b pb-2">
      <div className="flex gap-1">
        {tabs.map((t) => (
          <Link
            key={t.id}
            to={t.to}
            params={{ name }}
            className={`rounded px-3 py-1 text-sm ${
              active === t.id ? 'bg-blue-600 text-white' : 'hover:bg-gray-100'
            }`}
          >
            {t.label}
          </Link>
        ))}
      </div>
      <Link to="/" className="text-sm text-blue-600 hover:underline">
        ← exams
      </Link>
    </div>
  )
}
