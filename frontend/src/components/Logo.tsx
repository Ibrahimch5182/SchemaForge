import { useId } from "react";

export function LogoMark({ size = 30 }: { size?: number }) {
  const gid = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" focusable="false" className="logo-mark">
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ff5f3d" />
          <stop offset="1" stopColor="#ffb35c" />
        </linearGradient>
      </defs>
      <path d="M16 3.5 26.5 9.75v12.5L16 28.5 5.5 22.25V9.75z" fill="none" stroke={`url(#${gid})`} strokeWidth="2.2" strokeLinejoin="round" />
      <path d="m16 10.5 3.6 5.5-3.6 5.5-3.6-5.5z" fill={`url(#${gid})`} />
    </svg>
  );
}

export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <span className="logo">
      <LogoMark />
      {!compact && (
        <span className="logo-word">
          Schema<span>Forge</span>
        </span>
      )}
    </span>
  );
}
