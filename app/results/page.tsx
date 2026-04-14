import SectionHeader from "@/components/SectionHeader";

const metrics = [
  { name: "Accuracy",            desc: "Overall classification accuracy across 6 classes per scenario." },
  { name: "Macro F1-Score",      desc: "Unweighted average F1, accounts fairly for class imbalance." },
  { name: "Per-class F1",        desc: "F1 per kelas — No Finding, Infiltration, Effusion, Atelectasis, Nodule, Pneumothorax." },
  { name: "Robustness Drop",     desc: "Degradasi dari Scenario A (ideal) ke Scenario D (worst-case)." },
  { name: "Imbalance Δ",         desc: "Selisih performa Balanced vs Imbalanced untuk mengukur ketahanan class skew." },
  { name: "Noise Resilience",    desc: "Performa per severity level (1/2/3) untuk tiap jenis noise." },
];

const methods = ["MedSD (FE+FA)", "ConvNeXtV2", "DINOv2", "MaxViT"];
const scenarios = ["1 — Bal", "2 — Bal+FSA", "3 — Imbal", "4 — Imbal+FSA"];

export default function ResultsPage() {
  return (
    <main style={{ paddingTop: "56px" }}>
      <div style={{ background: "var(--ink)", padding: "52px 40px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "#6a5f50", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "14px" }}>ChestPrior · Results</div>
          <h1 style={{ fontFamily: "var(--font-serif)", fontWeight: 900, fontSize: "clamp(2rem,4vw,3rem)", color: "var(--paper)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
            Experimental Results
          </h1>
        </div>
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "72px 40px" }}>

        {/* Status notice */}
        <div style={{ padding: "24px 32px", border: "1px solid var(--accent-light)", background: "var(--accent-muted)", borderRadius: "4px", marginBottom: "64px", display: "flex", gap: "16px", alignItems: "flex-start" }}>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--accent)", marginTop: "1px" }}>⏳</span>
          <div>
            <div style={{ fontFamily: "var(--font-serif)", fontWeight: 600, fontSize: "1rem", color: "var(--ink)", marginBottom: "4px" }}>Experiments in Progress</div>
            <p style={{ fontSize: "0.85rem", color: "var(--ink-mid)", lineHeight: "1.6" }}>
              Halaman ini akan diperbarui dengan hasil aktual setelah semua skenario selesai dievaluasi. Placeholder TBD akan diganti dengan angka eksperimen yang sesungguhnya.
            </p>
          </div>
        </div>

        {/* Placeholder result table */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader index="1." label="Primary Results" title="Accuracy Across Scenarios" subtitle="Semua nilai TBD — ganti dengan hasil aktual setelah eksperimen selesai." />
          <div className="card" style={{ overflow: "hidden" }}>
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  {scenarios.map((s) => <th key={s}>{s}</th>)}
                  <th>Δ 1→4</th>
                </tr>
              </thead>
              <tbody>
                {methods.map((m, i) => (
                  <tr key={m}>
                    <td style={{ fontWeight: 500, color: "var(--ink)", fontFamily: "var(--font-mono)", fontSize: "0.78rem" }}>{m}</td>
                    {scenarios.map((s) => (
                      <td key={s} style={{ textAlign: "center", color: "var(--ink-faint)", fontStyle: "italic" }}>TBD</td>
                    ))}
                    <td style={{ textAlign: "center", color: "var(--ink-faint)", fontStyle: "italic" }}>TBD</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ marginTop: "10px", fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--ink-faint)" }}>
            * Reported as Accuracy (%) / Macro F1 (%)
          </p>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* Metrics */}
        <section>
          <SectionHeader index="2." label="Evaluation Metrics" title="What Will Be Reported" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px" }}>
            {metrics.map((m, i) => (
              <div className="card" key={m.name} style={{ padding: "22px" }}>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "var(--ink-faint)", marginBottom: "6px" }}>
                  {String(i + 1).padStart(2, "0")}
                </div>
                <h3 style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "0.95rem", color: "var(--ink)", marginBottom: "6px" }}>{m.name}</h3>
                <p style={{ fontSize: "0.82rem", color: "var(--ink-light)", lineHeight: "1.6" }}>{m.desc}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
