import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "@fontsource-variable/space-grotesk";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { readConfig } from "./api/config";
import { SchemaForgeClient } from "./api/client";
import App from "./App";
import { ApiProvider } from "./state/api-context";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/landing.css";
import "./styles/workspace.css";

const config = readConfig();
const api = new SchemaForgeClient({ baseUrl: config.apiBaseUrl, queryTimeoutMs: config.queryTimeoutMs });

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ApiProvider api={api}>
      <App />
    </ApiProvider>
  </StrictMode>,
);
