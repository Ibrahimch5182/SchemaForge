/**
 * Tiny, lossless SQL tokenizer for display. It returns typed text spans that
 * React renders as ordinary text nodes -- generated SQL is never parsed as
 * HTML and never executed here. `tokens.map(t => t.text).join("") === input`.
 */

export type TokenType = "keyword" | "function" | "string" | "number" | "comment" | "identifier" | "quoted" | "punct" | "space";

export interface SqlToken {
  type: TokenType;
  text: string;
}

const KEYWORDS = new Set(
  (
    "select from where group by order having limit offset distinct all as on using join inner left right full outer cross natural " +
    "union intersect except with recursive and or not in is null like glob between exists case when then else end asc desc " +
    "insert update delete create drop alter attach detach pragma values set into replace index table view trigger begin commit rollback " +
    "cast collate true false over partition window filter escape indexed match regexp"
  ).split(" "),
);

const isSpace = (c: string) => /\s/.test(c);
const isIdentStart = (c: string) => /[A-Za-z_]/.test(c);
const isIdentPart = (c: string) => /[A-Za-z0-9_$]/.test(c);
const isDigit = (c: string) => c >= "0" && c <= "9";

function readQuoted(src: string, i: number, quote: string): number {
  let j = i + 1;
  while (j < src.length) {
    if (src[j] === quote) {
      if (src[j + 1] === quote) {
        j += 2; // doubled quote is an escape
        continue;
      }
      return j + 1;
    }
    j++;
  }
  return src.length; // unterminated: consume the rest, still lossless
}

export function tokenizeSql(src: string): SqlToken[] {
  const out: SqlToken[] = [];
  let i = 0;
  const push = (type: TokenType, end: number) => {
    out.push({ type, text: src.slice(i, end) });
    i = end;
  };

  while (i < src.length) {
    const c = src[i]!;
    const next = src[i + 1];
    if (isSpace(c)) {
      let j = i + 1;
      while (j < src.length && isSpace(src[j]!)) j++;
      push("space", j);
    } else if (c === "-" && next === "-") {
      let j = i + 2;
      while (j < src.length && src[j] !== "\n") j++;
      push("comment", j);
    } else if (c === "/" && next === "*") {
      const end = src.indexOf("*/", i + 2);
      push("comment", end === -1 ? src.length : end + 2);
    } else if (c === "'") {
      push("string", readQuoted(src, i, "'"));
    } else if (c === '"' || c === "`") {
      push("quoted", readQuoted(src, i, c));
    } else if (c === "[") {
      const end = src.indexOf("]", i + 1);
      push("quoted", end === -1 ? src.length : end + 1);
    } else if (isDigit(c) || (c === "." && next !== undefined && isDigit(next))) {
      let j = i + 1;
      while (j < src.length && /[0-9.eE_]/.test(src[j]!)) {
        if ((src[j] === "e" || src[j] === "E") && (src[j + 1] === "+" || src[j + 1] === "-")) j++;
        j++;
      }
      push("number", j);
    } else if (isIdentStart(c)) {
      let j = i + 1;
      while (j < src.length && isIdentPart(src[j]!)) j++;
      const word = src.slice(i, j);
      if (KEYWORDS.has(word.toLowerCase())) {
        push("keyword", j);
      } else {
        let k = j;
        while (k < src.length && (src[k] === " " || src[k] === "\t")) k++;
        push(src[k] === "(" ? "function" : "identifier", j);
      }
    } else {
      push("punct", i + 1);
    }
  }
  return out;
}
