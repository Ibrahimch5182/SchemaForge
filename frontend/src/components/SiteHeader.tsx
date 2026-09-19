import type { ReactNode } from "react";
import { RouteLink, type Route } from "../lib/route";
import { ArrowRight } from "./icons";
import { Logo } from "./Logo";

interface Props {
  route: Route;
  navigate: (r: Route) => void;
  /** Right-aligned slot (e.g. the backend status pill in the workspace). */
  right?: ReactNode;
}

export function SiteHeader({ route, navigate, right }: Props) {
  return (
    <header className={`site-header ${route === "workspace" ? "is-app" : ""}`}>
      <div className="site-header-inner">
        <RouteLink to="landing" navigate={navigate} className="brand-link" aria-label="SchemaForge home">
          <Logo />
        </RouteLink>

        {route === "landing" ? (
          <nav className="site-nav" aria-label="Primary">
            <a href="#pipeline">How it works</a>
            <a href="#results">Results</a>
            <a href="#deployment">Deployment</a>
            <a href="#safety">Safety</a>
          </nav>
        ) : (
          <span className="crumb" aria-hidden="true">
            Workspace
          </span>
        )}

        <div className="site-header-right">
          {right}
          {route === "landing" && (
            <RouteLink to="workspace" navigate={navigate} className="btn btn-primary btn-sm">
              Open workspace <ArrowRight size={15} />
            </RouteLink>
          )}
        </div>
      </div>
    </header>
  );
}
