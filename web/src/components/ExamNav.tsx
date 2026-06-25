import { Link } from '@tanstack/react-router'

type Tab = 'grade' | 'import' | 'scheme' | 'results'

export function ExamNav({ name, active }: { name: string; active: Tab }) {
  const tabs: { id: Tab; label: string; to: string }[] = [
    { id: 'grade', label: 'Grade', to: '/session/$name' },
    { id: 'import', label: 'Import', to: '/session/$name/import' },
    { id: 'scheme', label: 'Scheme', to: '/session/$name/scheme' },
    { id: 'results', label: 'Results', to: '/session/$name/results' },
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
        ← sessions
      </Link>
    </div>
  )
}
