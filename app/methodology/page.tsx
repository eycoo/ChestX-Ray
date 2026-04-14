import SectionHeader from "@/components/SectionHeader";

const scenarios = [
  { id: "1", label: "Balanced · No Augmentation",      desc: "~416 sampel/kelas, tanpa augmentasi. Baseline ideal untuk mengukur kemampuan representasi fitur murni pada distribusi seimbang.",       tag: "Baseline",       col: "var(--accent-2)" },
  { id: "2", label: "Balanced · +FSA",                  desc: "~416 sampel/kelas + Feature Space Augmentation (Gaussian noise, feature dropout, Mixup). Uji apakah augmentasi di feature space meningkatkan generalisasi.", tag: "Augmented",  col: "var(--accent)" },
  { id: "3", label: "Imbalanced · No Augmentation",     desc: "Distribusi 10:1 (No Finding mendominasi), tanpa augmentasi. Simulasi dataset medis nyata dengan class imbalance. Class weights menangani skew.",          tag: "Imbalanced",     col: "var(--ink-mid)" },
  { id: "4", label: "Imbalanced · +FSA",                desc: "Distribusi 10:1 + Feature Space Augmentation. Worst-case — imbalance dikombinasikan dengan augmentasi untuk mengukur ketahanan setiap feature extractor.",  tag: "Hard",           col: "#b22222" },
];

export default function Methodology() {
  return (
    <main style={{ paddingTop: "56px" }}>
      {/* Page title band */}
      <div style={{ background: "var(--ink)", padding: "52px 40px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "#6a5f50", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "14px" }}>
            ChestPrior · Methodology
          </div>
          <h1 style={{ fontFamily: "var(--font-serif)", fontWeight: 900, fontSize: "clamp(2rem,4vw,3rem)", color: "var(--paper)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
            Pipeline &amp; Experimental Design
          </h1>
        </div>
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "72px 40px" }}>

        {/* 4 Scenarios */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader index="1." label="Experimental Design" title="Four Evaluation Scenarios" subtitle="Kombinasi dua dimensi: distribusi kelas (balanced/imbalanced) × augmentasi fitur (tanpa/+FSA)." />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            {scenarios.map((s) => (
              <div key={s.id} className="card" style={{ padding: "28px", borderTop: `3px solid ${s.col}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                  <span style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.1rem", color: "var(--ink)" }}>
                    Scenario {s.id}
                  </span>
                  <span className="chip chip-ink">{s.tag}</span>
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: s.col, fontWeight: 500, marginBottom: "10px" }}>{s.label}</div>
                <p style={{ fontSize: "0.85rem", color: "var(--ink-light)", lineHeight: "1.7" }}>{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* FSA Augmentation Detail */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader index="2." label="Augmentation Design" title="Feature Space Augmentation (FSA)" subtitle="Tiga teknik augmentasi diterapkan langsung di feature space (bukan pixel space) saat training." />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px" }}>
            {[
              { name: "Gaussian Noise", param: "σ = 0.01", desc: "Injeksi noise ringan pada vektor fitur untuk meningkatkan ketahanan terhadap variasi input." },
              { name: "Feature Dropout", param: "10% dims → 0", desc: "Secara acak mematikan 10% dimensi fitur untuk mendorong representasi yang lebih robust dan tidak over-rely pada fitur tertentu." },
              { name: "Mixup", param: "α = 0.2 (Beta)", desc: "Interpolasi konveks antara dua sampel dalam feature space dengan soft label — mengurangi overfitting dan meningkatkan generalisasi." },
            ].map((t) => (
              <div key={t.name} className="card" style={{ padding: "24px" }}>
                <div style={{ fontFamily: "var(--font-mono)", fontWeight: 500, fontSize: "0.85rem", color: "var(--accent)", marginBottom: "4px" }}>{t.name}</div>
                <div style={{ fontSize: "0.72rem", color: "var(--ink-faint)", fontFamily: "var(--font-mono)", marginBottom: "10px" }}>{t.param}</div>
                <p style={{ fontSize: "0.84rem", color: "var(--ink-light)", lineHeight: "1.65" }}>{t.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* FE+FA Pipeline — Architecture Diagram */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader index="3." label="Core Pipeline" title="Dual Feature Aggregation (FE+FA)" subtitle="Arsitektur end-to-end dari gambar X-ray mentah hingga representasi 128-dim siap klasifikasi." />

          {/* Architecture Block Diagram */}
          <div style={{ background: "var(--ink)", borderRadius: "8px", padding: "48px 40px", marginBottom: "28px" }}>
            {/* Top Flow: Input → VAE → Noise → U-Net */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0", flexWrap: "wrap", marginBottom: "32px" }}>
              {[
                { label: "X-Ray Image", sub: "512 \u00d7 512", bg: "var(--accent)" },
                { label: "VAE Encoder", sub: "Frozen", bg: "var(--ink-mid)" },
                { label: "Noise (t=10)", sub: "z₀ + ε → zₜ", bg: "var(--ink-mid)" },
                { label: "U-Net Forward", sub: "Frozen SD v1.4 + LoRA", bg: "var(--accent)" },
              ].map((b, i) => (
                <div key={b.label} style={{ display: "flex", alignItems: "center" }}>
                  <div style={{
                    background: b.bg, borderRadius: "6px", padding: "16px 20px",
                    textAlign: "center", minWidth: "140px",
                  }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "#fff", fontWeight: 600, marginBottom: "2px" }}>{b.label}</div>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.58rem", color: "rgba(255,255,255,0.6)" }}>{b.sub}</div>
                  </div>
                  {i < 3 && (
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.2rem", color: "var(--ink-faint)", padding: "0 6px" }}>→</div>
                  )}
                </div>
              ))}
            </div>

            {/* Dual outputs from U-Net */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", maxWidth: "600px", margin: "0 auto 32px" }}>
              <div style={{ background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "14px 18px", textAlign: "center" }}>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--accent-light)", marginBottom: "4px" }}>Feature Maps</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "rgba(255,255,255,0.5)" }}>F* [B, C, H, W] × 4 scales</div>
              </div>
              <div style={{ background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "6px", padding: "14px 18px", textAlign: "center" }}>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--accent-light)", marginBottom: "4px" }}>Attention Maps</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6rem", color: "rgba(255,255,255,0.5)" }}>{"{Aᵢ}"} [B, 1, H, W] saliency</div>
              </div>
            </div>
            <div style={{ textAlign: "center", fontFamily: "var(--font-mono)", fontSize: "1.2rem", color: "var(--ink-faint)", marginBottom: "24px" }}>↓</div>

            {/* FA Modules Row */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0", flexWrap: "wrap", marginBottom: "32px" }}>
              {[
                { label: "DFATB", sub: "Spatial + Channel Attn", color: "var(--accent-2)" },
                { label: "FAFN", sub: "Split-gate MLP", color: "var(--accent-2)" },
                { label: "Diff. Denoising", sub: "λ-weighted A₁ − A₂", color: "var(--accent-2)" },
              ].map((b, i) => (
                <div key={b.label} style={{ display: "flex", alignItems: "center" }}>
                  <div style={{
                    background: b.color, borderRadius: "6px", padding: "16px 22px",
                    textAlign: "center", minWidth: "150px",
                  }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "#fff", fontWeight: 600, marginBottom: "2px" }}>{b.label}</div>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.58rem", color: "rgba(255,255,255,0.6)" }}>{b.sub}</div>
                  </div>
                  {i < 2 && (
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.2rem", color: "var(--ink-faint)", padding: "0 6px" }}>→</div>
                  )}
                </div>
              ))}
            </div>
            <div style={{ textAlign: "center", fontFamily: "var(--font-mono)", fontSize: "1.2rem", color: "var(--ink-faint)", marginBottom: "24px" }}>↓</div>

            {/* Bottom: GAP → Bottleneck → MLP */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0", flexWrap: "wrap" }}>
              {[
                { label: "GAP + Concat", sub: "Global Avg Pool", bg: "var(--ink-mid)" },
                { label: "Bottleneck", sub: "→ z₁₂₈ [B, 128]", bg: "var(--ink-mid)" },
                { label: "MLP Head", sub: "→ 6 classes", bg: "var(--accent)" },
              ].map((b, i) => (
                <div key={b.label} style={{ display: "flex", alignItems: "center" }}>
                  <div style={{
                    background: b.bg, borderRadius: "6px", padding: "16px 22px",
                    textAlign: "center", minWidth: "140px",
                  }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "#fff", fontWeight: 600, marginBottom: "2px" }}>{b.label}</div>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.58rem", color: "rgba(255,255,255,0.6)" }}>{b.sub}</div>
                  </div>
                  {i < 2 && (
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.2rem", color: "var(--ink-faint)", padding: "0 6px" }}>→</div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Textual legend */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px" }}>
            {[
              { color: "var(--accent)", label: "Input / Output", desc: "Titik masuk dan keluar pipeline — model menerima gambar X-ray dan menghasilkan 6 logit kelas." },
              { color: "var(--accent-2)", label: "FA Modules (Trainable)", desc: "Komponen yang dilatih: DFATB, FAFN, dan Differential Denoising mengolah representasi U-Net." },
              { color: "var(--ink-mid)", label: "Frozen / Utility", desc: "Komponen yang dibekukan (VAE, U-Net) dan operasi utilitas (noise, GAP, bottleneck)." },
            ].map((l) => (
              <div key={l.label} style={{ display: "flex", gap: "10px", alignItems: "flex-start" }}>
                <div style={{ width: "12px", height: "12px", borderRadius: "3px", background: l.color, flexShrink: 0, marginTop: "4px" }} />
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", fontWeight: 500, color: "var(--ink)", marginBottom: "2px" }}>{l.label}</div>
                  <p style={{ fontSize: "0.8rem", color: "var(--ink-light)", lineHeight: "1.5" }}>{l.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* Generative Prior note — SDXL removed, single card */}
        <section>
          <SectionHeader index="4." label="Key Concept" title="Generative Prior as Feature Encoder" />
          <div className="card" style={{ padding: "28px", borderLeft: "4px solid var(--accent)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
              <h3 style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1rem", color: "var(--ink)" }}>Medical X-ray Stable Diffusion</h3>
              <span className="chip chip-accent">Proposed</span>
            </div>
            <p style={{ fontSize: "0.84rem", color: "var(--ink-light)", lineHeight: "1.7" }}>
              Fine-tuned Stable Diffusion pada domain chest X-ray (CompVis/stable-diffusion-v1-4 + LoRA). Memiliki prior yang lebih mendalam terhadap distribusi intensitas piksel X-ray, tekstur jaringan paru, dan pola patologis subtle. Feature maps multi-layer diekstrak dari U-Net denoiser yang dibekukan (frozen).
            </p>
          </div>
          <div style={{ marginTop: "20px", padding: "20px 24px", background: "var(--accent-muted)", border: "1px solid var(--accent-light)", borderRadius: "4px" }}>
            <p style={{ fontSize: "0.85rem", color: "var(--ink-mid)", lineHeight: "1.7" }}>
              <strong>Penting:</strong> Generative prior tidak digunakan untuk <em>menghasilkan</em> gambar baru. U-Net denoiser dibekukan (frozen) dan berfungsi sebagai <strong>feature encoder</strong> — representasi internal dari pre-training dimanfaatkan langsung untuk mengekstrak fitur dari gambar X-ray yang sudah ada.
            </p>
          </div>
        </section>
      </div>

      <style>{`@media (max-width: 768px) { .two-col { grid-template-columns: 1fr !important; } div[style*="grid-template-columns: 1fr 1fr 1fr"] { grid-template-columns: 1fr !important; } div[style*="grid-template-columns: 1fr 1fr"] { grid-template-columns: 1fr !important; } }`}</style>
    </main>
  );
}
