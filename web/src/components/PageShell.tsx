import type { ReactNode } from 'react'
import { Link } from '@tanstack/react-router'
import { ArrowLeft, type LucideIcon } from 'lucide-react'
import { Button } from '#/components/ui/button'
import { cn } from '#/lib/utils'

/**
 * Canonical page container. Every top-level route renders its content inside a
 * PageShell so width, centering, and padding stay identical across the app.
 *
 * Width/padding/centering are locked (appended last in `cn`, so they win the
 * twMerge conflict resolution): a caller's `className` can only adjust the inner
 * vertical rhythm, never reintroduce a per-page `max-w-*`. Divergent max-widths
 * are exactly the drift this prevents.
 */
export function PageShell({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div className={cn('space-y-6', className, 'mx-auto max-w-4xl p-8')}>{children}</div>
  )
}

/**
 * Standard top-of-page header: a back button to the exam list, an optional
 * leading icon, the title, and optional trailing actions. Used by the
 * standalone pages (the exam tabs use ExamNav instead).
 */
export function PageHeader({
  title,
  icon: Icon,
  actions,
}: {
  title: string
  icon?: LucideIcon
  actions?: ReactNode
}) {
  return (
    <div className="flex items-center gap-2">
      <Link to="/">
        <Button variant="ghost" size="icon" aria-label="Back">
          <ArrowLeft className="h-4 w-4" />
        </Button>
      </Link>
      <h1 className="flex items-center gap-2 text-2xl font-bold">
        {Icon ? <Icon className="h-6 w-6" /> : null}
        {title}
      </h1>
      {actions ? <div className="ml-auto flex items-center gap-2">{actions}</div> : null}
    </div>
  )
}
