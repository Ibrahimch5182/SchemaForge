export type ApiErrorKind =
  | "network" // could not reach the backend (offline, refused, CORS)
  | "timeout" // client-side wait elapsed
  | "aborted" // user cancelled
  | "http" // backend answered with a structured (or unstructured) non-2xx
  | "malformed"; // a response that does not match the contract

export interface ApiErrorInit {
  kind: ApiErrorKind;
  message: string;
  status?: number;
  /** Backend error code, e.g. `unknown_database`, `invalid_request`. */
  code?: string;
  requestId?: string | null;
  fields?: { field: string; issue: string }[];
}

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;
  readonly code?: string;
  readonly requestId?: string | null;
  readonly fields?: { field: string; issue: string }[];

  constructor(init: ApiErrorInit) {
    super(init.message);
    this.name = "ApiError";
    this.kind = init.kind;
    this.status = init.status;
    this.code = init.code;
    this.requestId = init.requestId;
    this.fields = init.fields;
  }
}
