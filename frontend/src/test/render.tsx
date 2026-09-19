import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import type { SchemaForgeApi } from "../api/client";
import { ApiProvider } from "../state/api-context";
import { Workspace } from "../pages/Workspace";
import { vi } from "vitest";

export function renderWithApi(ui: ReactElement, api: SchemaForgeApi) {
  const user = userEvent.setup();
  return { user, ...render(<ApiProvider api={api}>{ui}</ApiProvider>) };
}

export function renderWorkspace(api: SchemaForgeApi) {
  return renderWithApi(<Workspace navigate={vi.fn()} />, api);
}
