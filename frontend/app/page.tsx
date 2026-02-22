"use client";

import { useState, useRef, useCallback } from "react";

const API_URL = "http://107.20.32.82:8000/predict";

const MODEL_METRICS = [
  { value: "87.2%", label: "Accuracy" },
  { value: "0.42", label: "Loss (val)" },
  { value: "8", label: "Categorías" },
  { value: "~20k", label: "Imágenes (dataset)" },
];

interface PredictionResult {
  class: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export default function Home() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) return;
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
    setFile(f);
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const f = e.dataTransfer?.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleClassify = async () => {
    if (!file) {
      setError("Selecciona una imagen primero.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Error del servidor (${res.status})`);
      }

      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Error al clasificar la imagen."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const sortedProbs = result
    ? Object.entries(result.probabilities).sort(([, a], [, b]) => b - a)
    : [];
  const maxProb = result
    ? Math.max(...Object.values(result.probabilities))
    : 0;

  return (
    <div className="flex justify-center p-8">
      <div className="w-full max-w-[560px]">
        {/* Intro */}
        <div className="mb-8">
          <h1 className="text-[1.6rem] font-semibold tracking-tight text-white mb-1 leading-tight">
            Sistema de clasificación automática de edificaciones urbanas
          </h1>
          <p className="text-muted text-[0.95rem]">
            Sube una imagen de calle y obtén la categoría de la edificación
            usando deep learning (transfer learning con ResNet50).
          </p>
        </div>

        {/* Card 1 — Entrada */}
        <div className="card">
          <h2 className="text-base font-semibold text-[#b8c0ca] mb-3 flex items-center gap-2">
            <span className="text-accent font-bold">1</span> Entrada
          </h2>

          <div
            className={`upload-zone ${isDragOver ? "drag-over" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Vista previa"
                className="max-w-full max-h-60 rounded-[10px] object-contain"
              />
            ) : (
              <>
                <div className="text-[2.75rem] opacity-70">🏢</div>
                <p className="text-muted text-[0.9rem]">
                  Imagen RGB de una edificación urbana
                </p>
                <p className="text-[0.85rem] text-muted-dark">
                  Arrastra una imagen aquí o haz clic para seleccionar
                </p>
              </>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
          />

          <button
            onClick={handleClassify}
            disabled={isLoading}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? "Clasificando…" : "Clasificar imagen"}
          </button>

          {error && (
            <p className="mt-3 text-[0.85rem] text-red-400">{error}</p>
          )}
        </div>

        {/* Card 2 — Resultado */}
        {result && (
          <div className="card">
            <h2 className="text-base font-semibold text-[#b8c0ca] mb-3 flex items-center gap-2">
              <span className="text-accent font-bold">2</span> Resultado —
              Modelo (ResNet50, ImageNet)
            </h2>

            <p className="text-xl font-semibold text-green mb-4">
              Categoría predicha:{" "}
              <span className="text-[#c4c7cc] font-normal text-[0.95rem]">
                {result.label}
              </span>
            </p>

            <p className="text-muted text-[0.85rem] mb-3">
              Probabilidad por clase:
            </p>

            <div className="space-y-2">
              {sortedProbs.map(([name, prob]) => (
                <div
                  key={name}
                  className="flex items-center gap-3 text-[0.85rem]"
                >
                  <span className="w-[150px] text-muted">{name}</span>
                  <div className="flex-1 h-[22px] bg-white/[0.08] rounded-[6px] overflow-hidden">
                    <div
                      className="prob-bar"
                      style={{ width: `${(prob / maxProb) * 100}%` }}
                    />
                  </div>
                  <span className="w-11 text-right text-[#b8c0ca] font-medium">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Card 4 — Métricas */}
        {result && (
          <div className="card">
            <h2 className="text-base font-semibold text-[#b8c0ca] mb-3 flex items-center gap-2">
              Métricas del modelo
            </h2>
            <p className="text-muted text-[0.85rem] mb-2">
              Evaluación del desempeño (métricas estándar de clasificación):
            </p>
            <div className="grid grid-cols-[repeat(auto-fit,minmax(140px,1fr))] gap-4 mt-2">
              {MODEL_METRICS.map((m) => (
                <div
                  key={m.label}
                  className="bg-black/20 rounded-[10px] p-4 text-center"
                >
                  <div className="text-2xl font-bold text-green">{m.value}</div>
                  <div className="text-[0.8rem] text-muted-dark mt-1">
                    {m.label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-10 pt-6 border-t border-white/[0.08] text-[#6b7280] text-[0.85rem] text-center">
          <strong className="text-[#8b95a0]">
            Clasificador de edificaciones urbanas
          </strong>{" "}
          — Prototipo para análisis urbano, distribución logística y
          aplicaciones académicas.
        </div>
      </div>
    </div>
  );
}
