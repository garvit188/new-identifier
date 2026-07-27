import { useState, useRef, useEffect } from 'react'
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
  return new ort.Tensor('float32', pixels, [1, 224, 224, 3])
}

export default function App() {
  const [model, setModel] = useState(null)
  const [modelError, setModelError] = useState(null)
  const [imageURL, setImageURL] = useState(null)
  const [result, setResult] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const fileInputRef = useRef(null)
  const canvasRef = useRef(null)
  const imgRef = useRef(null)

  useEffect(() => {
    let dead = false
    ort.env.wasm.numThreads = 1
    ort.InferenceSession.create('/models/model.onnx')
      .then(m => { if (!dead) setModel(m) })
      .catch(e => { if (!dead) setModelError(e.message || String(e)) })
    return () => { dead = true }
  }, [])

  function handleFileSelect(e) {
    const file = e.target.files?.[0]
    if (!file) return
    if (imageURL) URL.revokeObjectURL(imageURL)
    setResult(null)
    setImageURL(URL.createObjectURL(file))
  }

  function handleDrop(e) {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (!file) return
    if (imageURL) URL.revokeObjectURL(imageURL)
    setResult(null)
    setImageURL(URL.createObjectURL(file))
  }

  function handleImgLoad() {
    const img = imgRef.current
    const cvs = canvasRef.current
    if (!img || !cvs) return
    cvs.width = 224
    cvs.height = 224
    cvs.getContext('2d').drawImage(img, 0, 0, 224, 224)
  }

  async function handleClassify() {
    if (!model || !imageURL || predicting) return
    setPredicting(true)
    setResult(null)
    handleImgLoad()
    await new Promise(r => setTimeout(r, 100))
    try {
      const tensor = preprocess(canvasRef.current)
      const feeds = {}
      feeds[model.inputNames[0]] = tensor
      const outputMap = await model.run(feeds)
      const outTensor = outputMap[model.outputNames[0]]
      const val = outTensor.data[0]
      const label = val > 0.5 ? 'DOG' : 'CAT'
      const conf = ((val > 0.5 ? val : 1 - val) * 100).toFixed(1)
      setResult({ label, conf })
    } catch (e) {
      console.error(e)
      setResult({ label: 'ERROR', conf: '0' })
    }
    setPredicting(false)
  }

  function handleReset() {
    if (imageURL) URL.revokeObjectURL(imageURL)
    setImageURL(null)
    setResult(null)
  }

  const showUpload = !imageURL

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      <header style={{ textAlign: 'center' }}>
        <h1 style={{
          fontFamily: "'DM Serif Display', serif",
          fontSize: 36, fontWeight: 400, lineHeight: 1.1,
          color: 'var(--ink)', marginBottom: 6, letterSpacing: '-0.02em'
        }}>
          dog or cat?
        </h1>
        <p style={{ color: 'var(--ink-muted)', fontSize: 14 }}>
          upload a photo, let the model decide
        </p>
      </header>

      {modelError && (
        <div style={{
          padding: 16, borderRadius: 12,
          background: '#fdf0ed', border: '1px solid #e8c8bd',
          fontSize: 13, color: '#7a3e2e', lineHeight: 1.5
        }}>
          model load error: {modelError}
        </div>
      )}

      {!model && !modelError && (
        <div style={{
          height: 180, borderRadius: 12,
          background: 'var(--surface)', border: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--ink-muted)', fontSize: 13
        }}>
          loading model…
        </div>
      )}

      {model && showUpload && (
        <div
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          style={{
            borderRadius: 12,
            border: '2px dashed var(--border)',
            background: 'var(--surface)',
            padding: '48px 24px',
            textAlign: 'center',
            cursor: 'pointer'
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
            stroke="var(--ink-faint)" strokeWidth="1.5"
            style={{ margin: '0 auto 10px', display: 'block' }}>
            <rect x="3" y="3" width="18" height="18" rx="3" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <path d="m21 15-5-5L5 21" />
          </svg>
          <p style={{ color: 'var(--ink-muted)', fontSize: 13 }}>
            tap to choose or drag an image
          </p>
        </div>
      )}

      {model && !showUpload && (
        <div style={{
          borderRadius: 12, overflow: 'hidden',
          border: '1px solid var(--border)',
          background: 'var(--surface)'
        }}>
          <img
            ref={imgRef}
            src={imageURL}
            alt="preview"
            onLoad={handleImgLoad}
            style={{ width: '100%', display: 'block' }}
          />
        </div>
      )}

      <canvas
        ref={canvasRef}
        width={224}
        height={224}
        style={{ position: 'fixed', left: '-9999px', top: 0, width: 0, height: 0, visibility: 'hidden' }}
      />

      {model && !showUpload && !predicting && !result && (
        <button
          onClick={handleClassify}
          style={{
            padding: '14px 0',
            border: 'none',
            borderRadius: 10,
            background: 'var(--accent)',
            color: '#fff',
            fontSize: 15,
            fontWeight: 600,
            cursor: 'pointer',
            letterSpacing: '-0.01em',
            width: '100%',
            boxShadow: '0 1px 4px rgba(181,103,58,0.3)'
          }}
        >
          classify
        </button>
      )}

      {predicting && (
        <div style={{
          padding: '14px 0', borderRadius: 10,
          background: 'var(--surface)', border: '1px solid var(--border)',
          textAlign: 'center', color: 'var(--ink-muted)', fontSize: 14
        }}>
          classifying…
        </div>
      )}

      {result && (
        <div style={{
          borderRadius: 12, background: 'var(--surface)',
          border: '1px solid var(--border)', padding: 28,
          textAlign: 'center', animation: 'fade-up 0.5s ease',
          boxShadow: 'var(--shadow)'
        }}>
          <div style={{ fontSize: 52, lineHeight: 1, marginBottom: 10 }}>
            {result.label === 'DOG' ? '🐕' : result.label === 'CAT' ? '🐈' : '—'}
          </div>
          <div style={{
            fontFamily: "'DM Serif Display', serif",
            fontSize: 28, letterSpacing: '-0.02em',
            color: result.label === 'DOG' ? 'var(--accent)' :
                   result.label === 'CAT' ? 'var(--gold)' : 'var(--ink-muted)',
            marginBottom: 16
          }}>
            {result.label.toLowerCase()}
          </div>
          {result.conf !== '0' && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 10,
              background: 'var(--bg)', borderRadius: 10, padding: '10px 16px'
            }}>
              <span style={{
                fontSize: 11, color: 'var(--ink-muted)', fontWeight: 500,
                textTransform: 'uppercase', letterSpacing: '0.04em'
              }}>
                confidence
              </span>
              <div style={{
                flex: 1, height: 5, borderRadius: 3,
                background: 'var(--border)', overflow: 'hidden'
              }}>
                <div style={{
                  width: result.conf + '%', height: '100%', borderRadius: 3,
                  background: result.label === 'DOG'
                    ? 'linear-gradient(90deg, var(--accent), #d48a5a)'
                    : 'linear-gradient(90deg, var(--gold), #c9a876)',
                  transition: 'width 0.8s ease'
                }} />
              </div>
              <span style={{
                fontSize: 14, fontWeight: 600, minWidth: 40,
                textAlign: 'right', color: 'var(--ink)'
              }}>
                {result.conf}%
              </span>
            </div>
          )}
          <button
            onClick={handleReset}
            style={{
              marginTop: 16, padding: '8px 24px',
              border: '1px solid var(--border)', borderRadius: 8,
              background: 'transparent', color: 'var(--ink-muted)',
              fontSize: 13, cursor: 'pointer'
            }}
          >
            try another
          </button>
        </div>
      )}

      <p style={{
        textAlign: 'center', color: 'var(--ink-faint)', fontSize: 11,
        letterSpacing: '0.02em', marginTop: 4
      }}>
        runs entirely in your browser · no data uploaded
      </p>
    </div>
  )
}
