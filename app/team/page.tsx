export default function TeamPage() {
  const team = [
    { name: "Syahribanun", role: "Lead Researcher", contrib: "Research design · pipeline development · FE+FA implementation · writeup", initials: "S" },
    { name: "Ahmad Naufal Farras", role: "Researcher", contrib: "Pengembangan pipeline Feature Extraction · Pengembangan dan Deployment Model di Web", initials: "A" },
  ];
  return (
    <main style={{ paddingTop: "56px" }}>
      <div style={{ background: "var(--ink)", padding: "52px 40px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "#6a5f50", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "14px" }}>ChestPrior · Team</div>
          <h1 style={{ fontFamily: "var(--font-serif)", fontWeight: 900, fontSize: "clamp(2rem,4vw,3rem)", color: "var(--paper)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
            Research Team
          </h1>
        </div>
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "72px 40px" }}>

        {/* Members */}
        <section style={{ marginBottom: "72px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {team.map((m) => (
              <div key={m.name} className="card" style={{ padding: "32px", display: "grid", gridTemplateColumns: "64px 1fr", gap: "28px", alignItems: "center" }}>
                <div style={{
                  width: "64px", height: "64px", borderRadius: "50%",
                  background: "var(--ink)", display: "flex", alignItems: "center", justifyContent: "center",
                  fontFamily: "var(--font-serif)", fontWeight: 900, fontSize: "1.4rem", color: "var(--paper)",
                }}>{m.initials}</div>
                <div>
                  <h3 style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.3rem", color: "var(--ink)", marginBottom: "4px" }}>{m.name}</h3>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--accent)", letterSpacing: "0.06em", marginBottom: "8px" }}>{m.role}</div>
                  <p style={{ fontSize: "0.84rem", color: "var(--ink-light)" }}>{m.contrib}</p>
                </div>
              </div>
            ))}
            {/* Placeholder */}
            <div style={{ padding: "20px 24px", border: "1px dashed var(--rule)", borderRadius: "4px", fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--ink-faint)", textAlign: "center" }}>
              Tambahkan anggota tim lain di <code>app/team/page.tsx</code>
            </div>
          </div>
        </section>

        <hr className="hr" style={{ marginBottom: "72px" }} />

        {/* Institution */}
        <section style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "40px" }}>
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "var(--accent)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "14px" }}>Institution</div>
            <h2 style={{ fontFamily: "var(--font-serif)", fontWeight: 700, fontSize: "1.3rem", color: "var(--ink)", marginBottom: "12px", letterSpacing: "-0.01em" }}>
              Institut Teknologi Sepuluh Nopember
            </h2>
            <p style={{ fontSize: "0.88rem", color: "var(--ink-light)", lineHeight: "1.8" }}>
              Departemen Teknik Informatika<br/>
              Laboratorium Komputasi Cerdas dan Visi (KCV)<br/>
              Surabaya, Jawa Timur, Indonesia
            </p>
          </div>
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "var(--accent)", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: "14px" }}>About This Project</div>
            <p style={{ fontSize: "0.88rem", color: "var(--ink-light)", lineHeight: "1.8" }}>
              Proyek final seleksi asisten lab KCV — menggabungkan penelitian generative model, computer vision, dan medical imaging untuk menjawab pertanyaan robustness yang belum banyak dieksplorasi dalam literatur.
            </p>
          </div>
        </section>
      </div>

      <style>{`@media (max-width: 768px) { section[style*="grid-template-columns: 1fr 1fr"] { grid-template-columns: 1fr !important; } }`}</style>
    </main>
  );
}
