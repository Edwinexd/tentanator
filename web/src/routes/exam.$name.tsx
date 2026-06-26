import { createFileRoute, Outlet } from '@tanstack/react-router'

// Layout for an exam: the grade/import/scheme/results/pdf tabs are child routes
// that render through this Outlet. Each child renders its own ExamNav.
export const Route = createFileRoute('/exam/$name')({ component: () => <Outlet /> })
