import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Final grade for display, noting the raw grade when the scheme's round_up lifted it. */
export function gradeLabel(s: { grade: string; rounded_from?: string | null }): string {
  if (!s.grade) return '-'
  return s.rounded_from ? `${s.grade} (rounded from ${s.rounded_from})` : s.grade
}
