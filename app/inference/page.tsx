"use client";

import { useState, useRef } from "react";
import SectionHeader from "@/components/SectionHeader";

export default function InferencePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState<string>("A chest X-ray");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const [backendStatus, setBackendStatus] = useState<string>("checking...");

  const checkBackend = async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendStatus(`${data.device.toUpperCase()} · ${data.warmup_done ? "Warm" : "Cold"}`);
      } else {
        setBackendStatus("unreachable");
      }
    } catch {
      setBackendStatus("offline");
    }
  };

  const handleWarmup = async () => {
    setBackendStatus("warming up...");
    try {
      const res = await fetch(`${API_URL}/warmup`);
      if (res.ok) {
        const data = await res.json();
        setBackendStatus(`${data.device.toUpperCase()} · Warm (${data.warmup_latency_s}s)`);
      }
    } catch {
      setBackendStatus("warmup failed");
    }
  };

  // Check backend on mount
  useState(() => { checkBackend(); });

  const handlePredict = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("prompt", prompt);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`API returned status: ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to communicate with inference server.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main style={{ paddingTop: "56px", minHeight: "100vh" }}>
      <div style={{ background: "var(--ink)", padding: "52px 40px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "#6a5f50", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "14px" }}>ChestPrior · Inference</div>
          <h1 style={{ fontFamily: "var(--font-serif)", fontWeight: 900, fontSize: "clamp(2rem,4vw,3rem)", color: "var(--paper)", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
            Interactive Demo
          </h1>
        </div>
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "72px 40px" }}>

        {/* Backend status bar */}
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "14px 24px", marginBottom: "32px", borderRadius: "6px",
          background: backendStatus.includes("Warm") ? "var(--accent-2-light)" : backendStatus.includes("offline") || backendStatus.includes("unreachable") ? "#f8d7d7" : "var(--paper-dark)",
          border: `1px solid ${backendStatus.includes("Warm") ? "#a0ccbf" : backendStatus.includes("offline") || backendStatus.includes("unreachable") ? "#e0a0a0" : "var(--rule)"}`,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div style={{
              width: "8px", height: "8px", borderRadius: "50%",
              background: backendStatus.includes("Warm") ? "var(--accent-2)" : backendStatus.includes("offline") || backendStatus.includes("unreachable") ? "#b22222" : "var(--accent)",
              animation: backendStatus.includes("...") ? "pulse 1s infinite" : "none",
            }} />
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--ink-mid)" }}>
              Backend: {backendStatus}
            </span>
          </div>
          <div style={{ display: "flex", gap: "8px" }}>
            <button onClick={checkBackend} style={{
              fontFamily: "var(--font-mono)", fontSize: "0.7rem", padding: "6px 14px",
              border: "1px solid var(--rule)", borderRadius: "3px", background: "var(--paper)",
              color: "var(--ink-mid)", cursor: "pointer",
            }}>Refresh</button>
            <button onClick={handleWarmup} style={{
              fontFamily: "var(--font-mono)", fontSize: "0.7rem", padding: "6px 14px",
              border: "none", borderRadius: "3px", background: "var(--accent)",
              color: "#fff", cursor: "pointer", fontWeight: 500,
            }}>Warm up GPU</button>
          </div>
        </div>

        <section style={{ marginBottom: "72px" }}>
          <SectionHeader index="1." label="Input" title="Upload X-Ray" subtitle="Upload a single Chest X-Ray image to test the hybrid feature extraction and classification pipeline in real-time." />
          
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "40px" }}>
            {/* Input Panel */}
            <div className="card" style={{ padding: "32px", display: "flex", flexDirection: "column", gap: "24px" }}>
              
              <div>
                <label style={{ display: "block", fontFamily: "var(--font-mono)", fontSize: "0.75rem", letterSpacing: "0.08em", color: "var(--ink-mid)", marginBottom: "8px" }}>IMAGE UPLOAD</label>
                <div 
                  onClick={() => fileInputRef.current?.click()}
                  style={{ 
                    border: "2px dashed var(--rule)", borderRadius: "6px", padding: "40px 20px", 
                    textAlign: "center", cursor: "pointer", background: "var(--paper-dark)",
                    transition: "all 0.2s ease"
                  }}
                >
                  <p style={{ fontFamily: "var(--font-sans)", fontSize: "0.9rem", color: "var(--ink-light)" }}>
                    {selectedFile ? selectedFile.name : "Click here to upload an X-Ray (.png, .jpg)"}
                  </p>
                  <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" style={{ display: "none" }} />
                </div>
              </div>

              <div>
                <label style={{ display: "block", fontFamily: "var(--font-mono)", fontSize: "0.75rem", letterSpacing: "0.08em", color: "var(--ink-mid)", marginBottom: "8px" }}>TEXT PROMPT</label>
                <input 
                  type="text" 
                  value={prompt} 
                  onChange={(e) => setPrompt(e.target.value)}
                  style={{ 
                    width: "100%", padding: "12px", border: "1px solid var(--rule)", 
                    borderRadius: "4px", background: "var(--paper)", fontFamily: "var(--font-sans)",
                    fontSize: "0.9rem", color: "var(--ink)"
                  }} 
                />
              </div>

              <button 
                onClick={handlePredict}
                disabled={!selectedFile || isLoading}
                style={{
                  background: !selectedFile || isLoading ? "var(--rule)" : "var(--accent)",
                  color: !selectedFile || isLoading ? "var(--ink-faint)" : "#fff",
                  border: "none", padding: "14px", borderRadius: "4px", fontSize: "0.95rem",
                  fontFamily: "var(--font-sans)", fontWeight: 600, cursor: !selectedFile || isLoading ? "not-allowed" : "pointer",
                  transition: "background 0.2s ease"
                }}
              >
                {isLoading ? "Running Prediction..." : "Run Classification"}
              </button>

              {error && (
                <div style={{ color: "#b22222", fontSize: "0.85rem", marginTop: "8px", fontFamily: "var(--font-mono)" }}>
                  {error}
                </div>
              )}
            </div>

            {/* Preview & Result Panel */}
            <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
              {previewUrl && (
                 <div style={{ border: "1px solid var(--rule)", padding: "8px", borderRadius: "4px", background: "#fff", display: "inline-block", width: "fit-content" }}>
                    <img src={previewUrl} alt="Preview" style={{ objectFit: "contain", maxHeight: "250px" }} />
                 </div>
              )}

              {result && (
                <div className="card" style={{ padding: "32px", borderLeft: "4px solid var(--accent)" }}>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--ink-faint)", marginBottom: "4px" }}>PREDICTION RESULT</div>
                  <h2 style={{ fontFamily: "var(--font-serif)", fontSize: "2rem", fontWeight: 700, margin: "0 0 4px 0", color: "var(--ink)" }}>
                    {result.prediction}
                  </h2>
                  <p style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--accent)", margin: "0 0 24px 0" }}>
                    Confidence: {(result.confidence * 100).toFixed(2)}%
                  </p>

                  <div style={{ borderTop: "1px solid var(--rule)", paddingTop: "16px" }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.08em", color: "var(--ink-mid)", marginBottom: "12px" }}>PROBABILITIES</div>
                    {Object.entries(result.probabilities).sort(([,a], [,b]) => (b as number) - (a as number)).map(([className, prob]: [string, any]) => (
                      <div key={className} style={{ marginBottom: "12px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                          <span style={{ fontSize: "0.8rem", color: "var(--ink)", fontWeight: 500 }}>{className}</span>
                          <span style={{ fontSize: "0.75rem", color: "var(--ink-light)", fontFamily: "var(--font-mono)" }}>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{ width: "100%", height: "6px", background: "var(--rule)", borderRadius: "3px", overflow: "hidden" }}>
                          <div style={{ width: `${Math.max(prob * 100, 1)}%`, height: "100%", background: className === result.prediction ? "var(--accent)" : "var(--ink-light)", transition: "width 0.4s ease" }}></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            
          </div>
        </section>
      </div>

      <style>{`@media (max-width: 768px) { div[style*="grid-template-columns: 1fr 1fr"] { grid-template-columns: 1fr !important; } }`}</style>
    </main>
  );
}
