import { useState, useRef, useEffect, useCallback } from 'react'
import * as ort from 'onnxruntime-web'

function preprocess(canvas) {
  const ctx = canvas.getContext('2d')
  const { data } = ctx.getImageData(0, 0, 224, 224)
  const pixels = new Float32Array(224 * 224 * 3)
  for (let i = 0; i < 224 * 224; i++) {
    pixels[i * 3]     = data[i * 4] / 255
    pixels[i * 3 + 1] = data[i * 4 + 1] / 255
    pixels[i * 3 + 2] = data[i * 4 + 2] / 255
  }
  return new ort.Tensor(Float32Array, pixels, [1, 224, 224, 3])
}

export default function App() {
  const [model, setModel] = useState(null)
  const [loading, setLoading] = useState(true)
  const [image, setImage] = useState(null)
  const [result, setResult] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef(null)
  const canvasRef = useRef(null)
  const imgRef = useRef(null)

  useEffect(() => {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/'
    ort.InferenceSession.create('/models/model.onnx')
      .then(setModel)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const drawToCanvas = useCallback(() => {
    const img = imgRef.current
    const cvs = canvasRef.current
    if (!img || !cvs) return
    cvs.width = 224
    cvs.height = 224
    cvs.getContext('2d').drawImage(img, 0, 0, 224, 224)
  }, [])

  function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) return
    setResult(null)
    const url = URL.createObjectURL(file)
    setImage(url)
  }

  function onPick(e) {
    loadFile(e.target.files?.[0])
    e.target.value = ''
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    loadFile(e.dataTransfer.files?.[0])
  }

  async function predict() {
    if (!model || !image) return
    setPredicting(true)
    drawToCanvas()
    await new Promise(r => requestAnimationFrame(r))
    try {
      const tensor = preprocess(canvasRef.current)
      const feeds = { [model.inputNames[0]]: tensor }
      const { data } = await model.run(feeds)
      const val = data[model.outputNames[0]][0]
      const label = val > 0.5 ? 'DOG' : 'CAT'
      const confidence = ((val > 0.5 ? val : 1 - val) * 100).toFixed(1)
      setResult({ label, confidence, value: val })
    } catch {
      setResult({ label: 'ERROR', confidence: '—' })
    } finally {
      setPredicting(false)
    }
  }

  const droppable = !predicting && !loading

  return (
    <div style={{
      maxWidth: 440, margin: '0 auto',
      display: 'flex', flexDirection: 'column', gap: 32
    }}>
      <header style={{ textAlign: 'center' }}>
        <h1 style={{
          fontFamily: "'Plus Jakarta Sans', sans-serif",
          fontSize: 32, fontWeight: 800, letterSpacing: '-0.03em',
          marginBottom: 8, color: 'var(--ink)'
        }}>
          dog or cat?
        </h1>
        <p style={{ color: 'var(--ink-muted)', fontSize: 15, lineHeight: 1.5 }}>
          Drop a photo and let the model decide
        </p>
      </header>

      {loading ? (
        <div style={{
          height: 320, borderRadius: 'var(--radius)',
          background: 'var(--surface)', border: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--ink-faint)', fontSize: 14
        }}>
          loading model…
        </div>
      ) : (
        <>
          <div
            onDragOver={e => { e.preventDefault(); if (droppable) setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
            style={{
              borderRadius: 'var(--radius)',
              border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border)'}`,
              background: dragOver ? 'var(--accent-subtle)' : 'var(--surface)',
              padding: image ? 0 : 48,
              textAlign: 'center', cursor: 'pointer',
              transition: 'var(--transition)',
              minHeight: image ? undefined : 200,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              position: 'relative', overflow: 'hidden'
            }}
          >
            <input ref={fileRef} type="file" accept="image/*" onChange={onPick} hidden />

            {!image && (
              <div>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--ink-faint)" strokeWidth="1.5" style={{ margin: '0 auto 12px', display: 'block' }}>
                  <rect x="3" y="3" width="18" height="18" rx="3" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <path d="m21 15-5-5L5 21" />
                </svg>
                <p style={{ color: 'var(--ink-muted)', fontSize: 14 }}>tap or drag here to upload</p>
              </div>
            )}

            {image && (
              <img ref={imgRef} src={image} alt=""
                onLoad={drawToCanvas}
                style={{ width: '100%', display: 'block' }}
              />
            )}
          </div>

          <canvas ref={canvasRef} aria-hidden="true"
            style={{ position: 'absolute', left: '-9999px', top: 0, width: 224, height: 224 }}
          />

          {image && (
            <div style={{ display: 'flex', gap: 10, justifyContent: 'center' }}>
              <button onClick={predict} disabled={predicting}
                style={{
                  padding: '14px 40px', border: 'none', borderRadius: 'var(--radius-sm)',
                  background: predicting ? 'var(--ink-faint)' : 'var(--accent)',
                  color: '#fff', fontSize: 15, fontWeight: 600,
                  cursor: predicting ? 'not-allowed' : 'pointer',
                  transition: 'var(--transition)',
                  width: '100%', maxWidth: 240,
                  letterSpacing: '-0.01em'
                }}
                onMouseEnter={e => { if (!predicting) e.target.style.transform = 'scale(1.02)' }}
                onMouseLeave={e => e.target.style.transform = 'scale(1)'}
              >
                {predicting ? 'classifying…' : 'classify'}
              </button>
            </div>
          )}

          {result && (
            <div style={{
              borderRadius: 'var(--radius)',
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              padding: 28, textAlign: 'center',
              animation: 'fadeUp 0.5s ease'
            }}>
              <div style={{
                fontSize: 48, lineHeight: 1, marginBottom: 12,
                filter: result.label === 'ERROR' ? 'grayscale(1)' : undefined
              }}>
                {result.label === 'DOG' ? '🐕' : result.label === 'CAT' ? '🐈' : '—'}
              </div>
              <div style={{
                fontFamily: "'Plus Jakarta Sans', sans-serif",
                fontSize: 26, fontWeight: 800, letterSpacing: '-0.02em',
                color: result.label === 'DOG' ? 'var(--accent)' :
                       result.label === 'CAT' ? 'var(--accent-gold)' : 'var(--ink-muted)',
                marginBottom: 4,
                textTransform: 'lowercase'
              }}>
                {result.label === 'DOG' ? 'dog' : result.label === 'CAT' ? 'cat' : 'error'}
              </div>
              {result.confidence !== '—' && (
                <div style={{
                  marginTop: 16, display: 'flex', alignItems: 'center', gap: 10,
                  justifyContent: 'center'
                }}>
                  <div style={{
                    flex: 1, height: 4, borderRadius: 2,
                    background: 'var(--border)', maxWidth: 160, overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${result.confidence}%`, height: '100%',
                      borderRadius: 2,
                      background: result.label === 'DOG'
                        ? 'linear-gradient(90deg, var(--accent), #d48a5a)'
                        : 'linear-gradient(90deg, var(--accent-gold), #c9a876)',
                      transition: 'width 0.8s ease'
                    }} />
                  </div>
                  <span style={{ color: 'var(--ink-muted)', fontSize: 13, fontWeight: 500, minWidth: 36, textAlign: 'right' }}>
                    {result.confidence}%
                  </span>
                </div>
              )}
              <button onClick={() => { setImage(null); setResult(null) }}
                style={{
                  marginTop: 20, padding: '8px 20px', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)', background: 'transparent',
                  color: 'var(--ink-muted)', fontSize: 13, cursor: 'pointer',
                  transition: 'var(--transition)'
                }}
                onMouseEnter={e => { e.target.style.background = 'var(--surface-hover)'; e.target.style.borderColor = 'var(--ink-faint)' }}
                onMouseLeave={e => { e.target.style.background = 'transparent'; e.target.style.borderColor = 'var(--border)' }}
              >
                try another
              </button>
            </div>
          )}
        </>
      )}

      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}
