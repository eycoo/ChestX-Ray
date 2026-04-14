import SectionHeader from "@/components/SectionHeader";

const normalData = [
  // Scenario 1
  { s: 1, model: "DINOv2", acc: "33.33%", f1: "32.61%", auc: "71.00%" },
  { s: 1, model: "MaxViT", acc: "24.80%", f1: "24.54%", auc: "60.80%" },
  { s: 1, model: "ConvNeXtV2", acc: "30.40%", f1: "29.37%", auc: "68.36%" },
  { s: 1, model: "MedSD (FE+FA)", acc: "44.27%", f1: "43.90%", auc: "76.95%" },
  // Scenario 2
  { s: 2, model: "DINOv2", acc: "32.27%", f1: "31.91%", auc: "70.97%" },
  { s: 2, model: "MaxViT", acc: "23.73%", f1: "23.10%", auc: "63.43%" },
  { s: 2, model: "ConvNeXtV2", acc: "33.33%", f1: "32.98%", auc: "68.45%" },
  { s: 2, model: "MedSD (FE+FA)", acc: "45.33%", f1: "45.17%", auc: "77.47%" },
  // Scenario 3
  { s: 3, model: "DINOv2", acc: "37.60%", f1: "23.78%", auc: "67.71%" },
  { s: 3, model: "MaxViT", acc: "33.87%", f1: "24.94%", auc: "64.26%" },
  { s: 3, model: "ConvNeXtV2", acc: "29.87%", f1: "24.81%", auc: "68.43%" },
  { s: 3, model: "MedSD (FE+FA)", acc: "54.13%", f1: "38.55%", auc: "76.02%" },
  // Scenario 4
  { s: 4, model: "DINOv2", acc: "61.07%", f1: "24.78%", auc: "68.27%" },
  { s: 4, model: "MaxViT", acc: "52.80%", f1: "25.72%", auc: "63.84%" },
  { s: 4, model: "ConvNeXtV2", acc: "55.47%", f1: "18.61%", auc: "68.71%" },
  { s: 4, model: "MedSD (FE+FA)", acc: "59.47%", f1: "30.43%", auc: "78.78%" },
];

const corruptData = [
  // Scenario 1
  { s: 1, model: "DINOv2", acc: "25.87%", f1: "25.56%", auc: "64.70%" },
  { s: 1, model: "MaxViT", acc: "21.60%", f1: "21.18%", auc: "56.89%" },
  { s: 1, model: "ConvNeXtV2", acc: "28.80%", f1: "28.45%", auc: "63.87%" },
  { s: 1, model: "MedSD (FE+FA)", acc: "36.53%", f1: "35.99%", auc: "72.23%" },
  // Scenario 2
  { s: 2, model: "DINOv2", acc: "27.47%", f1: "26.41%", auc: "65.22%" },
  { s: 2, model: "MaxViT", acc: "22.93%", f1: "21.97%", auc: "57.22%" },
  { s: 2, model: "ConvNeXtV2", acc: "26.93%", f1: "26.65%", auc: "64.21%" },
  { s: 2, model: "MedSD (FE+FA)", acc: "38.67%", f1: "38.22%", auc: "72.72%" },
  // Scenario 3
  { s: 3, model: "DINOv2", acc: "49.07%", f1: "24.32%", auc: "65.63%" },
  { s: 3, model: "MaxViT", acc: "41.60%", f1: "22.81%", auc: "62.73%" },
  { s: 3, model: "ConvNeXtV2", acc: "39.20%", f1: "27.57%", auc: "64.00%" },
  { s: 3, model: "MedSD (FE+FA)", acc: "34.40%", f1: "26.63%", auc: "67.60%" },
  // Scenario 4
  { s: 4, model: "DINOv2", acc: "59.73%", f1: "23.06%", auc: "65.49%" },
  { s: 4, model: "MaxViT", acc: "49.07%", f1: "21.01%", auc: "63.45%" },
  { s: 4, model: "ConvNeXtV2", acc: "57.33%", f1: "22.71%", auc: "65.45%" },
  { s: 4, model: "MedSD (FE+FA)", acc: "59.20%", f1: "24.19%", auc: "68.84%" },
];

const scenarios = [
  { id: 1, name: "Scenario 1: Balanced · No Augmentation" },
  { id: 2, name: "Scenario 2: Balanced · +FSA" },
  { id: 3, name: "Scenario 3: Imbalanced · No Augmentation" },
  { id: 4, name: "Scenario 4: Imbalanced · +FSA" },
];

function ResultTable({ data, scenarioId }: { data: any[], scenarioId: number }) {
  const filtered = data.filter(d => d.s === scenarioId);
  return (
    <div className="card" style={{ overflow: "hidden", marginBottom: "20px" }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ background: "var(--paper)", borderBottom: "2px solid var(--ink)" }}>
            <th style={{ padding: "10px", textAlign: "left", fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-mid)" }}>Model</th>
            <th style={{ padding: "10px", textAlign: "right", fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-mid)" }}>Acc</th>
            <th style={{ padding: "10px", textAlign: "right", fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-mid)" }}>F1 (Macro)</th>
            <th style={{ padding: "10px", textAlign: "right", fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--ink-mid)" }}>AUC (OvR)</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((row, i) => {
            const isProposed = row.model.includes("MedSD");
            return (
              <tr key={row.model} style={{ 
                borderBottom: i === filtered.length - 1 ? "none" : "1px solid var(--rule)",
                background: isProposed ? "rgba(196, 98, 45, 0.05)" : "#fff" 
              }}>
                <td style={{ padding: "10px", fontSize: "0.85rem", fontWeight: isProposed ? 700 : 500, color: isProposed ? "var(--accent)" : "var(--ink)" }}>{row.model}</td>
                <td style={{ padding: "10px", textAlign: "right", fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: "var(--ink-mid)" }}>{row.acc}</td>
                <td style={{ padding: "10px", textAlign: "right", fontFamily: "var(--font-mono)", fontSize: "0.8rem", fontWeight: isProposed ? 700 : 400, color: isProposed ? "var(--accent)" : "var(--ink-mid)" }}>{row.f1}</td>
                <td style={{ padding: "10px", textAlign: "right", fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: "var(--ink-mid)" }}>{row.auc}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

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

          <h3 style={{ fontFamily: "var(--font-serif)", fontSize: "1.2rem", fontWeight: 700, marginBottom: "24px", marginTop: "40px", color: "var(--ink)" }}>
            Confusion Matrices (Normal Dataset)
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            {scenarios.map(sc => (
              <div key={`cm-normal-${sc.id}`} style={{ background: "var(--paper-dark)", padding: "20px", borderRadius: "6px" }}>
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
                          <img src={`/charts/cm-normal-s${sc.id}-${model.toLowerCase()}.png`} style={{ width: "100%", height: "100%", objectFit: "cover" }} alt={`CM Normal ${model} S${sc.id}`} />
                       </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
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
