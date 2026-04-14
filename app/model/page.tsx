import SectionHeader from "@/components/SectionHeader";

const extractors = [
  { abbr:"MedSD",  name:"Medical X-ray Stable Diffusion",  type:"Generative Prior",    paradigm:"Diffusion UNet encoder, domain fine-tuned on chest X-ray. Paired with full FE+FA (DFATB + FAFN + Differential Transformer) → z₁₂₈.", primary:true },
  { abbr:"CNX",    name:"ConvNeXtV2",                      type:"CNN",                 paradigm:"Modern ConvNet with FCMAE self-supervised pre-training. Captures rich local features with high computational efficiency.", primary:false },
  { abbr:"DINO",   name:"DINOv2",                          type:"Vision Transformer",  paradigm:"ViT trained via self-supervised knowledge distillation on LVD-142M. Strong global representations; transferable across domains.", primary:false },
  { abbr:"MXVT",   name:"MaxViT",                          type:"Hybrid",              paradigm:"Multi-Axis ViT combining local window + global grid attention with convolution. Captures local and global features simultaneously.", primary:false },
];

export default function ModelPage() {
  return (
    <main style={{ paddingTop: "56px" }}>
      <div style={{ background: "var(--ink)", padding: "52px 40px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "#6a5f50", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "14px" }}>ChestPrior · Model</div>
          <h1 style={{ fontFamily: "var(--font-serif)", fontWeight: 900, fontSize: "clamp(2rem,4vw,3rem)", color: "var(--paper)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
            Feature Extractors &amp; Architecture
          </h1>
        </div>
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "72px 40px" }}>

        {/* All extractors */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader index="1." label="Compared Methods" title="Four Feature Extractors" subtitle="Semua menghasilkan z₁₂₈ yang masuk ke MLP classifier identik — fair comparison terisolasi pada kualitas representasi fitur." />
          <div style={{ display: "flex", flexDirection: "column" }}>
            {extractors.map((e, i) => (
              <div key={e.abbr} style={{
                display: "grid", gridTemplateColumns: "56px 100px 140px 1fr",
                gap: "20px", alignItems: "start",
                padding: "24px 0",
                borderBottom: i < extractors.length - 1 ? "1px solid var(--rule)" : "none",
              }}>
                <span style={{ fontFamily: "var(--font-serif)", fontWeight: 900, fontSize: "1.6rem", color: "var(--paper-darker)", lineHeight: 1, paddingTop: "4px" }}>
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", fontWeight: 500, fontSize: "0.85rem", color: e.primary ? "var(--accent)" : "var(--ink)", marginBottom: "4px" }}>{e.abbr}</div>
                  {e.primary && <span className="chip chip-accent" style={{ fontSize: "0.58rem" }}>Proposed</span>}
                </div>
                <div>
                  <div style={{ fontFamily: "var(--font-sans)", fontSize: "0.88rem", fontWeight: 500, color: "var(--ink)", marginBottom: "4px", lineHeight: "1.3" }}>{e.name}</div>
                  <span className={e.primary ? "chip chip-accent" : "chip chip-ink"} style={{ fontSize: "0.6rem" }}>{e.type}</span>
                </div>
                <p style={{ fontSize: "0.85rem", color: "var(--ink-light)", lineHeight: "1.7", paddingTop: "2px" }}>{e.paradigm}</p>
              </div>
            ))}
          </div>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* FE+FA modules */}
        <section style={{ marginBottom: "72px" }}>
          <SectionHeader index="2." label="FE+FA Detail" title="Dual Feature Aggregation Modules" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px" }}>
            {[
              { abbr:"DFATB", full:"Dual Feature Aggregation Transformer Block", desc:"Menggabungkan spatial attention dan channel attention secara komplementer dari multi-layer feature maps U-Net denoiser." },
              { abbr:"FAFN",  full:"Feature Aggregation Feed-Forward Network",   desc:"Menangkap informasi spasial non-linear dan mengurangi redundansi channel dari representasi fitur yang telah diagregasi." },
              { abbr:"DiffTF",full:"Differential Denoising Transformer",         desc:"Menekan noise pada attention features melalui operasi diferensial antara dua independent attention maps." },
            ].map((m) => (
              <div key={m.abbr} className="card" style={{ padding: "24px" }}>
                <div style={{ fontFamily: "var(--font-mono)", fontWeight: 500, fontSize: "0.85rem", color: "var(--accent)", marginBottom: "4px" }}>{m.abbr}</div>
                <div style={{ fontSize: "0.72rem", color: "var(--ink-faint)", fontFamily: "var(--font-mono)", marginBottom: "10px", lineHeight: "1.4" }}>{m.full}</div>
                <p style={{ fontSize: "0.84rem", color: "var(--ink-light)", lineHeight: "1.65" }}>{m.desc}</p>
              </div>
            ))}
          </div>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* MLP */}
        <section>
          <SectionHeader index="3." label="Classifier" title="Unified MLP Head" subtitle="Arsitektur identik digunakan untuk semua 4 extractor di semua 4 skenario." />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", alignItems: "start" }}>
            <p style={{ fontSize: "0.9rem", color: "var(--ink-light)", lineHeight: "1.8" }}>
              Seluruh feature extractor menghasilkan representasi z₁₂₈ [B, 128] yang diumpankan ke MLP classifier yang sama persis. Desain ini memastikan bahwa perbedaan performa antar metode semata-mata mencerminkan kualitas representasi fitur, bukan keuntungan arsitektur classifier.
            </p>
            <div style={{ background: "var(--paper-dark)", border: "1px solid var(--rule)", borderRadius: "4px", padding: "24px", fontFamily: "var(--font-mono)", fontSize: "0.78rem", color: "var(--ink-mid)", lineHeight: "2.1" }}>
              <div style={{ color: "var(--ink-faint)", marginBottom: "4px" }}>// MLP Classifier</div>
              <div>Input:  z₁₂₈  [B, 128]</div>
              <div style={{ paddingLeft: "16px", color: "var(--ink-light)" }}>↓  LayerNorm(128)</div>
              <div style={{ paddingLeft: "16px", color: "var(--ink-light)" }}>↓  Linear(128 → 512)  +  GELU</div>
              <div style={{ paddingLeft: "16px", color: "var(--ink-light)" }}>↓  Dropout(0.3)</div>
              <div style={{ paddingLeft: "16px", color: "var(--ink-light)" }}>↓  Linear(512 → 6)</div>
              <div>Output: logits  [B, 6]</div>
            </div>
          </div>
        </section>
      </div>

      <style>{`@media (max-width: 768px) { div[style*="grid-template-columns: 56px"] { grid-template-columns: 1fr !important; } div[style*="grid-template-columns: 1fr 1fr 1fr"] { grid-template-columns: 1fr !important; } div[style*="grid-template-columns: 1fr 1fr"] { grid-template-columns: 1fr !important; } }`}</style>
    </main>
  );
}
