/** Minimal History-API router: two routes do not justify a routing dependency. */
import { useCallback, useEffect, useState, type AnchorHTMLAttributes, type MouseEvent, type ReactNode } from "react";

export type Route = "landing" | "workspace";

export const routeFromPath = (path: string): Route => (path.replace(/\/+$/, "") === "/workspace" ? "workspace" : "landing");
export const pathForRoute = (route: Route): string => (route === "workspace" ? "/workspace" : "/");

export function useRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(() => routeFromPath(window.location.pathname));
  useEffect(() => {
    const onPop = () => setRoute(routeFromPath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const navigate = useCallback((r: Route) => {
    const path = pathForRoute(r);
    if (window.location.pathname !== path) window.history.pushState({}, "", path);
    setRoute(r);
    window.scrollTo({ top: 0 });
  }, []);
  return [route, navigate];
}

interface RouteLinkProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href" | "onClick"> {
  to: Route;
  navigate: (r: Route) => void;
  children: ReactNode;
}

/** A real `<a href>` (middle-click / copy-link work) that navigates without a reload. */
export function RouteLink({ to, navigate, children, ...rest }: RouteLinkProps) {
  const onClick = (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    navigate(to);
  };
  return (
    <a href={pathForRoute(to)} onClick={onClick} {...rest}>
      {children}
    </a>
  );
}
