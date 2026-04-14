import Link from "next/link";

const classes = [
  { code: "No Finding",    n: 416,  risk: "Normal",    riskColor: "var(--accent-2)" },
  { code: "Infiltration",  n: 417,  risk: "Moderate",  riskColor: "var(--ink-mid)" },
  { code: "Effusion",      n: 416,  risk: "Moderate",  riskColor: "var(--ink-mid)" },
  { code: "Atelectasis",   n: 416,  risk: "Moderate",  riskColor: "var(--ink-mid)" },
  { code: "Nodule",        n: 418,  risk: "Suspicious", riskColor: "var(--accent)" },
  { code: "Pneumothorax",  n: 417,  risk: "Critical",  riskColor: "#b22222" },
];

const methods = [
  { abbr: "MedSD", full: "Medical X-ray Stable Diffusion + FE+FA", type: "Generative Prior" },
  { abbr: "CNX",   full: "ConvNeXtV2",                            type: "CNN" },
  { abbr: "DINO",  full: "DINOv2",                                type: "ViT" },
  { abbr: "MXVT",  full: "MaxViT",                                type: "Hybrid" },
];

export default function Home() {
  return (
    <main style={{ paddingTop: "56px" }}>

      {/* ── HERO ─────────────────────────────────────────────── */}
      <section style={{
        padding: "80px 40px 60px",
        maxWidth: "1100px", margin: "0 auto",
        display: "grid", gridTemplateColumns: "1fr 340px", gap: "60px", alignItems: "start",
      }}>
        <div className="fade-up">
          {/* Category label */}
          <div style={{
            display: "inline-flex", alignItems: "center", gap: "8px",
            fontFamily: "var(--font-mono)", fontSize: "0.65rem",
            letterSpacing: "0.14em", textTransform: "uppercase",
            color: "var(--accent)", marginBottom: "28px",
          }}>
            <span style={{ width: "24px", height: "1px", background: "var(--accent)", display: "inline-block" }}/>
            KCV Final Project · 2026
          </div>

          <h1 style={{
            fontFamily: "var(--font-serif)", fontWeight: 900,
            fontSize: "clamp(2.2rem, 4.5vw, 3.6rem)",
            lineHeight: "1.08", letterSpacing: "-0.03em",
            color: "var(--ink)", marginBottom: "28px",
          }}>
            Generative Priors<br/>
            <em style={{ fontWeight: 400, color: "var(--ink-mid)" }}>for Robust</em><br/>
            Chest X-ray<br/>Classification
          </h1>

          <p style={{
            fontSize: "1rem", lineHeight: "1.8",
            color: "var(--ink-light)", maxWidth: "520px", marginBottom: "36px",
          }}>
            Evaluasi komparatif generative prior berbasis diffusion model dan modul Dual Feature Aggregation untuk klasifikasi chest X-ray di bawah kondisi data imbalanced dan corrupt.
          </p>

          <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
            <Link href="/inference" style={{
              textDecoration: "none", padding: "11px 28px",
              background: "var(--accent)", color: "#fff",
              fontFamily: "var(--font-sans)", fontSize: "0.85rem", fontWeight: 500,
              borderRadius: "3px", letterSpacing: "0.01em",
            }}>Try Inference Model</Link>
            <Link href="/methodology" style={{
              textDecoration: "none", padding: "11px 28px",
              background: "var(--ink)", color: "var(--paper)",
              fontFamily: "var(--font-sans)", fontSize: "0.85rem", fontWeight: 500,
              borderRadius: "3px", letterSpacing: "0.01em",
            }}>Read Methodology</Link>
            <Link href="/results" style={{
              textDecoration: "none", padding: "11px 28px",
              border: "1px solid var(--rule)", color: "var(--ink-mid)",
              fontFamily: "var(--font-sans)", fontSize: "0.85rem",
              borderRadius: "3px",
            }}>View Results</Link>
          </div>
        </div>

        {/* Right sidebar — quick facts */}
        <div className="fade-up-2" style={{
          borderLeft: "2px solid var(--ink)",
          paddingLeft: "28px",
          paddingTop: "8px",
        }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", letterSpacing: "0.12em", color: "var(--ink-faint)", textTransform: "uppercase", marginBottom: "20px" }}>
            Quick Facts
          </div>
          {[
            ["Dataset",      "ChestX-ray14 (Wang et al., 2017)"],
            ["Subset",       "2,500 images sampled"],
            ["Classes",      "6 disease categories"],
            ["Scenarios",    "4 (Balanced/Imbalanced × No Aug/+FSA)"],
            ["Methods",      "4 feature extractors compared"],
            ["Augmentation", "Gaussian noise + Feature dropout + Mixup"],
          ].map(([k, v]) => (
            <div key={k} style={{ marginBottom: "14px" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "var(--ink-faint)", letterSpacing: "0.08em", marginBottom: "2px" }}>{k}</div>
              <div style={{ fontFamily: "var(--font-sans)", fontSize: "0.84rem", color: "var(--ink-mid)", lineHeight: "1.4" }}>{v}</div>
            </div>
          ))}
        </div>
      </section>

      {/* thin full-width rule */}
      <hr className="hr" style={{ margin: "0 40px" }} />

      {/* ── ABSTRACT ─────────────────────────────────────────── */}
      <section style={{ padding: "72px 40px", maxWidth: "1100px", margin: "0 auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "48px", alignItems: "start" }}>
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", letterSpacing: "0.14em", color: "var(--accent)", textTransform: "uppercase", marginBottom: "8px" }}>§ Abstract</div>
            <hr className="hr-accent" style={{ width: "32px" }} />
          </div>
          <div>
            <p className="pull-quote" style={{ marginBottom: "24px" }}>
              "Seberapa robust sebuah model klasifikasi chest X-ray ketika dihadapkan pada data yang imbalanced, corrupt, atau keduanya sekaligus?"
            </p>
            <p style={{ fontSize: "0.95rem", lineHeight: "1.85", color: "var(--ink-mid)", marginBottom: "16px" }}>
              Chest X-ray adalah modalitas pencitraan medis paling umum untuk deteksi penyakit toraks. Walaupun deep learning telah mencapai performa setara radiologis, pengujian selama ini dilakukan di kondisi ideal — jauh dari realitas klinis yang sering menghadirkan distribusi kelas tidak seimbang dan kualitas gambar yang terdegradasi.
            </p>
            <p style={{ fontSize: "0.95rem", lineHeight: "1.85", color: "var(--ink-mid)" }}>
              Penelitian ini mengeksplorasi pemanfaatan <strong>generative prior</strong> dari diffusion model (Medical X-ray SD) sebagai feature extractor, dikombinasikan dengan modul <strong>Dual Feature Aggregation (FE+FA)</strong> — DFATB, FAFN, dan Differential Transformer — yang mengagregasikan representasi multi-layer menjadi vektor 128-dim untuk klasifikasi.
            </p>
          </div>
        </div>
      </section>

      <hr className="hr" style={{ margin: "0 40px" }} />

      {/* ── DATASET ──────────────────────────────────────────── */}
      <section style={{ padding: "72px 40px", maxWidth: "1100px", margin: "0 auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "48px", alignItems: "start" }}>
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", letterSpacing: "0.14em", color: "var(--accent)", textTransform: "uppercase", marginBottom: "8px" }}>§ Dataset</div>
            <hr className="hr-accent" style={{ width: "32px" }} />
          </div>
          <div>
            <h2 style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.8rem", letterSpacing: "-0.02em", marginBottom: "8px" }}>
              ChestX-ray14
            </h2>
            <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--ink-faint)", marginBottom: "28px" }}>
              Wang et al., 2017 — 112,120 images · 30,805 patients · 14 disease labels
            </p>

            {/* Class table */}
            <div className="card" style={{ overflow: "hidden", marginBottom: "20px" }}>
              <table>
                <thead>
                  <tr>
                    <th>Disease Class</th>
                    <th>Balanced (n)</th>
                    <th>Risk Level</th>
                  </tr>
                </thead>
                <tbody>
                  {classes.map((c) => (
                    <tr key={c.code}>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.82rem", color: "var(--ink)" }}>{c.code}</td>
                      <td>{c.n}</td>
                      <td>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: c.riskColor, fontWeight: 500 }}>{c.risk}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p style={{ fontSize: "0.82rem", color: "var(--ink-light)", lineHeight: "1.7" }}>
              Skenario <em>imbalanced</em> menggunakan rasio 10:1 — No Finding mendominasi sementara Pneumothorax paling jarang. Skenario <em>balanced</em> menggunakan ~416 sampel per kelas.
            </p>
          </div>
        </div>
      </section>

      <hr className="hr" style={{ margin: "0 40px" }} />

      {/* ── METHODS ──────────────────────────────────────────── */}
      <section style={{ padding: "72px 40px", maxWidth: "1100px", margin: "0 auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "48px" }}>
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", letterSpacing: "0.14em", color: "var(--accent)", textTransform: "uppercase", marginBottom: "8px" }}>§ Methods</div>
            <hr className="hr-accent" style={{ width: "32px" }} />
          </div>
          <div>
            <h2 style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.8rem", letterSpacing: "-0.02em", marginBottom: "24px" }}>
              4 Feature Extractors Compared
            </h2>
            <div style={{ display: "flex", flexDirection: "column", gap: "1px" }}>
              {methods.map((m, i) => (
                <div key={m.abbr} style={{
                  display: "grid", gridTemplateColumns: "48px 100px 1fr auto",
                  gap: "16px", alignItems: "center",
                  padding: "16px 20px",
                  background: i % 2 === 0 ? "#fff" : "var(--paper-dark)",
                  borderLeft: i === 0 ? "3px solid var(--accent)" : "3px solid var(--rule)",
                }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--ink-faint)" }}>#{i + 1}</span>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.78rem", fontWeight: 500, color: i === 0 ? "var(--accent)" : "var(--ink)" }}>{m.abbr}</span>
                  <span style={{ fontSize: "0.85rem", color: "var(--ink-mid)" }}>{m.full}</span>
                  <span className={i === 0 ? "chip chip-accent" : "chip chip-ink"}>{m.type}</span>
                </div>
              ))}
            </div>
            <p style={{ marginTop: "16px", fontSize: "0.82rem", color: "var(--ink-light)", lineHeight: "1.7" }}>
              Semua extractor menghasilkan representasi 128-dim yang dimasukkan ke MLP classifier identik — perbandingan terisolasi pada kualitas representasi fitur.
            </p>
          </div>
        </div>
      </section>

      <hr className="hr" style={{ margin: "0 40px" }} />

      {/* ── CTA strip ────────────────────────────────────────── */}
      <section style={{ padding: "64px 40px", maxWidth: "1100px", margin: "0 auto" }}>
        <div style={{
          background: "var(--ink)", borderRadius: "4px",
          padding: "52px 60px",
          display: "grid", gridTemplateColumns: "1fr auto", alignItems: "center", gap: "40px",
        }}>
          <div>
            <h2 style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.7rem", color: "var(--paper)", letterSpacing: "-0.02em", marginBottom: "8px" }}>
              Explore the Full Paper
            </h2>
            <p style={{ color: "#a09080", fontSize: "0.88rem" }}>
              Methodology, model architecture, results, and references.
            </p>
          </div>
          <div style={{ display: "flex", gap: "10px" }}>
            <Link href="/methodology" style={{
              textDecoration: "none", padding: "10px 22px",
              background: "var(--accent)", color: "#fff",
              fontFamily: "var(--font-sans)", fontSize: "0.82rem", fontWeight: 500,
              borderRadius: "3px",
            }}>Methodology</Link>
            <Link href="/team" style={{
              textDecoration: "none", padding: "10px 22px",
              border: "1px solid #3d3528", color: "#a09080",
              fontFamily: "var(--font-sans)", fontSize: "0.82rem",
              borderRadius: "3px",
            }}>Team</Link>
          </div>
        </div>
      </section>

      <style>{`
        @media (max-width: 768px) {
          section > div[style*="grid-template-columns: 1fr 340px"],
          section > div[style*="grid-template-columns: 160px 1fr"],
          section > div[style*="grid-template-columns: 1fr auto"] {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </main>
  );
}
