/**
 * Tailwind class name helper used by every shadcn/ui component.
 * Combines clsx (conditional class joining) with tailwind-merge (which
 * deduplicates conflicting tailwind utilities so e.g. `px-2 px-4` collapses
 * to `px-4`).
 */
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
