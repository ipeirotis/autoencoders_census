/**
 * useCurrentUser - React Query hook that exposes the current authenticated
 * user (or null if logged out) plus helpers to refresh after login/logout.
 *
 * Centralizing auth state in a query lets every component reactively re-render
 * when the user logs in or out without prop drilling, and lets us cache the
 * /api/auth/me result so we don't refetch on every component mount.
 */

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getCurrentUser, type User } from "@/utils/auth";

export const CURRENT_USER_QUERY_KEY = ["currentUser"] as const;

export function useCurrentUser() {
  const queryClient = useQueryClient();

  const query = useQuery<User | null>({
    queryKey: CURRENT_USER_QUERY_KEY,
    queryFn: getCurrentUser,
    // Don't keep retrying on 401 - getCurrentUser already converts that to
    // null, so any thrown error here is a real failure (5xx, network) and
    // a single retry is enough.
    retry: 1,
    // Auth state rarely changes mid-session; refetch only on explicit
    // window focus to catch logouts in other tabs.
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: true,
  });

  /**
   * Force a refetch of /api/auth/me. Used after login/signup to swap the
   * cached null for the new user, and after logout to swap the cached user
   * for null.
   */
  const refresh = () => {
    return queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
  };

  /**
   * Imperatively set the cached user. Useful right after login/signup so
   * the UI swaps to the authenticated view immediately without waiting for
   * a network round-trip back to /api/auth/me.
   */
  const setUser = (user: User | null) => {
    queryClient.setQueryData(CURRENT_USER_QUERY_KEY, user);
  };

  return {
    user: query.data ?? null,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refresh,
    setUser,
  };
}
