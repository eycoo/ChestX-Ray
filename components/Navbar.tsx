"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Overview" },
  { href: "/methodology", label: "Methodology" },
  { href: "/results", label: "Results" },
  { href: "/team", label: "Team" },
  { href: "/inference", label: "Try Model" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", fn);
    return () => window.removeEventListener("scroll", fn);
  }, []);

  return (
    <nav style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
      height: "56px",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 40px",
      background: scrolled ? "rgba(245,240,232,0.97)" : "var(--paper)",
      borderBottom: scrolled ? "1px solid var(--rule)" : "1px solid transparent",
      backdropFilter: "blur(10px)",
      transition: "all 0.3s ease",
    }}>
      {/* Wordmark */}
      <Link href="/" style={{ textDecoration: "none", display: "flex", alignItems: "baseline", gap: "6px" }}>
        <span style={{
          fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.1rem",
          color: "var(--ink)", letterSpacing: "-0.01em",
        }}>ChestPrior</span>
        <span style={{
          fontFamily: "var(--font-mono)", fontSize: "0.6rem",
          color: "var(--ink-light)", letterSpacing: "0.06em",
        }}>2026</span>
      </Link>

      {/* Nav links */}
      <div style={{ display: "flex", gap: "2px" }}>
        {links.map((l) => {
          const active = pathname === l.href;
          return (
            <Link key={l.href} href={l.href} style={{
              textDecoration: "none",
              padding: "5px 14px",
              fontFamily: "var(--font-sans)",
              fontSize: "0.82rem",
              fontWeight: active ? 500 : 400,
              color: active ? "var(--accent)" : "var(--ink-mid)",
              borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
              transition: "all 0.15s ease",
              letterSpacing: "0.01em",
            }}>{l.label}</Link>
          );
        })}
      </div>

      {/* Right: institution tag */}
      <span style={{
        fontFamily: "var(--font-mono)", fontSize: "0.62rem",
        color: "var(--ink-faint)", letterSpacing: "0.08em",
      }}>ITS · KCV</span>
    </nav>
  );
}
