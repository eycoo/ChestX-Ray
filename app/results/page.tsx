import SectionHeader from "@/components/SectionHeader";

const metrics = [
  { name: "Accuracy",            desc: "Overall classification accuracy across 6 classes per scenario." },
  { name: "Macro F1-Score",      desc: "Unweighted average F1, accounts fairly for class imbalance." },
  { name: "Per-class F1",        desc: "F1 per kelas — No Finding, Infiltration, Effusion, Atelectasis, Nodule, Pneumothorax." },
  { name: "Robustness Drop",     desc: "Degradasi dari Scenario A (ideal) ke Scenario D (worst-case)." },
  { name: "Imbalance Δ",         desc: "Selisih performa Balanced vs Imbalanced untuk mengukur ketahanan class skew." },
  { name: "Noise Resilience",    desc: "Performa per severity level (1/2/3) untuk tiap jenis noise." },
];

const methods = ["MedSD (FE+FA)", "SDXL (FE+FA)", "ConvNeXtV2", "DINOv2", "MaxViT"];
const scenarios = ["A — Bal+Clean", "B — Imbal+Clean", "C — Bal+Corrupt", "D — Imbal+Corrupt"];

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

        {/* ==================================================== */}
        {/* NORMAL DATASET SECTION */}
        {/* ==================================================== */}
        <section style={{ marginBottom: "80px" }}>
          <SectionHeader index="1." label="Primary Results" title="Normal Dataset Performance" subtitle="Evaluation under pure conditions without induced noise or degradations." />
          
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px 32px", marginBottom: "40px" }}>
            {scenarios.map(sc => (
              <div key={sc.id}>
                <h3 style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--ink)", marginBottom: "12px", borderLeft: "3px solid var(--accent)", paddingLeft: "10px", fontWeight: 600 }}>
                  {sc.name}
                </h3>
                <ResultTable data={normalData} scenarioId={sc.id} />
              </div>
            ))}
          </div>

          <h3 style={{ fontFamily: "var(--font-serif)", fontSize: "1.2rem", fontWeight: 700, marginBottom: "16px", color: "var(--ink)" }}>
            Visualizations (Normal Dataset)
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "24px" }}>
            <div className="card" style={{ padding: "16px", display: "flex", flexDirection: "column", alignItems: "center" }}>
               <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-faint)", marginBottom: "12px" }}>PERFORMANCE BY SCENARIO (BARCHART)</div>
               <img src="/charts/barchart-normal.png" style={{ width: "100%", height: "auto", objectFit: "contain", borderRadius: "4px" }} alt="Barchart Normal" />
            </div>
            <div className="card" style={{ padding: "16px", display: "flex", flexDirection: "column", alignItems: "center" }}>
               <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-faint)", marginBottom: "12px" }}>AVG METRICS (RADAR)</div>
               <img src="/charts/radar-normal.png" style={{ width: "100%", height: "auto", objectFit: "contain", borderRadius: "4px" }} alt="Radar Chart Normal" />
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
                  <th>Δ A→D</th>
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
        </section>

        <hr className="hr-thick" style={{ marginBottom: "80px" }} />

        {/* ==================================================== */}
        {/* CORRUPT DATASET SECTION */}
        {/* ==================================================== */}
        <section style={{ marginBottom: "80px" }}>
          <SectionHeader index="2." label="Robustness Results" title="Corrupt Dataset Performance" subtitle="Evaluation under corrupted conditions (induced noise/blur) to measure resilience." />
          
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px 32px", marginBottom: "40px" }}>
            {scenarios.map(sc => (
              <div key={sc.id}>
                <h3 style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--ink-mid)", marginBottom: "12px", borderLeft: "3px solid #b22222", paddingLeft: "10px", fontWeight: 600 }}>
                  {sc.name}
                </h3>
                <ResultTable data={corruptData} scenarioId={sc.id} />
              </div>
            ))}
          </div>

          <h3 style={{ fontFamily: "var(--font-serif)", fontSize: "1.2rem", fontWeight: 700, marginBottom: "16px", color: "var(--ink)" }}>
            Visualizations (Corrupt Dataset)
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "24px" }}>
            <div className="card" style={{ padding: "16px", display: "flex", flexDirection: "column", alignItems: "center" }}>
               <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-faint)", marginBottom: "12px" }}>PERFORMANCE BY SCENARIO (BARCHART)</div>
               <img src="/charts/barchart-corrupt.png" style={{ width: "100%", height: "auto", objectFit: "contain", borderRadius: "4px" }} alt="Barchart Corrupt" />
            </div>
            <div className="card" style={{ padding: "16px", display: "flex", flexDirection: "column", alignItems: "center" }}>
               <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-faint)", marginBottom: "12px" }}>AVG METRICS (RADAR)</div>
               <img src="/charts/radar-corrupt.png" style={{ width: "100%", height: "auto", objectFit: "contain", borderRadius: "4px" }} alt="Radar Chart Corrupt" />
            </div>
          </div>

          <h3 style={{ fontFamily: "var(--font-serif)", fontSize: "1.2rem", fontWeight: 700, marginBottom: "24px", marginTop: "40px", color: "var(--ink)" }}>
            Confusion Matrices (Corrupt Dataset)
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            {scenarios.map(sc => (
              <div key={`cm-corrupt-${sc.id}`} style={{ background: "var(--paper-dark)", padding: "20px", borderRadius: "6px" }}>
                <h4 style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--ink-mid)", marginBottom: "16px", letterSpacing: "0.05em", fontWeight: 600 }}>
                  {sc.name.toUpperCase()}
                </h4>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "16px" }}>
                  {["MedSD", "ConvNeXtV2", "DINOv2", "MaxViT"].map(model => (
                    <div key={model} style={{ background: "#fff", border: "1px dashed var(--rule)", padding: "12px", borderRadius: "4px", textAlign: "center" }}>
                       <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: model === "MedSD" ? "var(--accent)" : "var(--ink)", marginBottom: "8px", fontWeight: model === "MedSD" ? 700 : 500 }}>
                         {model}
                       </div>
                       <div style={{ aspectRatio: "1/1", width: "100%", background: "rgba(0,0,0,0.02)", display: "flex", alignItems: "center", justifyContent: "center", border: "1px solid var(--paper-darker)", borderRadius: "2px", overflow: "hidden" }}>
                          <img src={`/charts/cm-corrupt-s${sc.id}-${model.toLowerCase()}.png`} style={{ width: "100%", height: "100%", objectFit: "cover" }} alt={`CM Corrupt ${model} S${sc.id}`} />
                       </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

      </div>
      <style>{`
        @media (max-width: 768px) { 
          div[style*="grid-template-columns: 1fr 1fr"] { grid-template-columns: 1fr !important; } 
          div[style*="grid-template-columns: 2fr 1fr"] { grid-template-columns: 1fr !important; } 
          div[style*="grid-template-columns: repeat(4, 1fr)"] { grid-template-columns: 1fr 1fr !important; } 
        }
      `}</style>
    </main>
  );
}
