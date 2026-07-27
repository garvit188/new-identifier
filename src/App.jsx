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
  const [modelError, setModelError] = useState(null)
  const [ready, setReady] = useState(false)
  const [image, setImage] = useState(null)
  const [result, setResult] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [errMsg, setErrMsg] = useState(null)
  const fileRef = useRef(null)
  const canvasRef = useRef(null)
  const imgRef = useRef(null)

  useEffect(() => {
    let dead = false
    async function init() {
      try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/'
        const m = await ort.InferenceSession.create('/models/model.onnx', { executionProviders: ['wasm'] })
        if (dead) return
        setModel(m)
        setReady(true)
      } catch (e) {
        if (dead) return
        console.error('model load failed:', e)
        setModelError(e.message || 'failed to load model')
      }
    }
    init()
    return () => { dead = true }
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
    setErrMsg(null)
    if (!file) return
    if (!file.type.startsWith('image/')) { setErrMsg('not an image'); return }
    setResult(null)
    const url = URL.createObjectURL(file)
    setImage(url)
  }

  function onPick(e) {
    const f = e.target.files?.[0]
    if (f) loadFile(f)
    e.target.value = ''
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    loadFile(e.dataTransfer.files?.[0])
  }

  async function predict() {
    if (!model || !image || predicting) return
    setPredicting(true)
    setResult(null)
    drawToCanvas()
    await new Promise(r => requestAnimationFrame(r))
    try {
      const tensor = preprocess(canvasRef.current)
      const feeds = { [model.inputNames[0]]: tensor }
      const outputMap = await model.run(feeds)
      const outName = model.outputNames[0]
      const outputData = outputMap[outName].data
      const val = outputData[0]
      const label = val > 0.5 ? 'DOG' : 'CAT'
      const confidence = ((val > 0.5 ? val : 1 - val) * 100).toFixed(1)
      setResult({ label, confidence })
    } catch (e) {
      console.error('predict error:', e)
      setErrMsg('prediction failed. check console.')
    } finally {
      setPredicting(false)
    }
  }

  function tryAgain() {
    URL.revokeObjectURL(image || '')
    setImage(null)
    setResult(null)
    setErrMsg(null)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>

      {/* HEADER */}
      <header style={{ textAlign: 'center' }}>
        <h1 style={{
          fontFamily: "'DM Serif Display', serif",
          fontSize: 38, fontWeight: 400, lineHeight: 1.1,
          color: 'var(--ink)', marginBottom: 8, letterSpacing: '-0.02em'
        }}>
          dog or cat?
        </h1>
        <p style={{ color: 'var(--ink-muted)', fontSize: 14, lineHeight: 1.5 }}>
          upload a photo, let the model decide
        </p>
      </header>

      {/* MODEL LOADING / ERROR */}
      {!ready && !modelError && (
        <div style={{
          height: 240, borderRadius: 'var(--radius)',
          background: 'var(--surface)', border: '1px solid var(--border)',
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', gap: 12
        }}>
          <div style={{
            width: 20, height: 20, border: '2px solid var(--border)',
            borderTopColor: 'var(--accent)', borderRadius: '50%',
            animation: 'spin 0.8s linear infinite'
          }} />
          <span style={{ color: 'var(--ink-muted)', fontSize: 13 }}>loading model…</span>
        </div>
      )}

      {modelError && (
        <div style={{
          padding: 20, borderRadius: 'var(--radius)',
          background: '#fdf0ed', border: '1px solid #e8c8bd',
          textAlign: 'center', fontSize: 13, color: '#7a3e2e', lineHeight: 1.5
        }}>
          {modelError}. check that{' '}
          <code style={{ background: '#f5e0d8', padding: '1px 6px', borderRadius: 4, fontSize: 12 }}>
            /models/model.onnx
          </code>{' '}exists.
        </div>
      )}

      {/* DROP ZONE / IMAGE PREVIEW */}
      {ready && (
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
          style={{
            borderRadius: 'var(--radius)',
            border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border)'}`,
            background: dragOver ? '#f5ede4' : 'var(--surface)',
            cursor: 'pointer', overflow: 'hidden',
            transition: `background var(--duration) var(--ease), border-color var(--duration) var(--ease)`,
            position: 'relative', minHeight: image ? undefined : 180,
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}
        >
          <input ref={fileRef} type="file" accept="image/*" onChange={onPick} hidden />

          {!image && !errMsg && (
            <div style={{ padding: '40px 24px', textAlign: 'center' }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--ink-faint)" strokeWidth="1.5"
                style={{ margin: '0 auto 10px', display: 'block' }}>
                <rect x="3" y="3" width="18" height="18" rx="3" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <path d="m21 15-5-5L5 21" />
              </svg>
              <p style={{ color: 'var(--ink-muted)', fontSize: 13 }}>tap to choose or drag here</p>
            </div>
          )}

          {!image && errMsg && (
            <div style={{ padding: '40px 24px', textAlign: 'center' }}>
              <p style={{ color: '#b5673a', fontSize: 13 }}>{errMsg}</p>
              <p style={{ color: 'var(--ink-faint)', fontSize: 12, marginTop: 6 }}>tap to try again</p>
            </div>
          )}

          {image && (
            <img ref={imgRef} src={image} alt=""
              onLoad={drawToCanvas}
              style={{ width: '100%', display: 'block', borderRadius: 'var(--radius-sm)' }}
            />
          )}
        </div>
      )}

      {/* HIDDEN CANVAS */}
      <canvas ref={canvasRef} aria-hidden="true"
        style={{ position: 'fixed', left: '-9999px', top: 0, width: 224, height: 224 }}
      />

      {/* CLASSIFY BUTTON */}
      {ready && image && !predicting && !result && (
        <button onClick={predict}
          style={{
            padding: '14px 0', border: 'none', borderRadius: 'var(--radius-sm)',
            background: 'var(--accent)', color: '#fff', fontSize: 15, fontWeight: 600,
            cursor: 'pointer', transition: `transform var(--duration) var(--ease), background var(--duration) var(--ease)`,
            letterSpacing: '-0.01em', width: '100%',
            boxShadow: '0 1px 4px rgba(181,103,58,0.3)'
          }}
          onMouseEnter={e => e.target.style.transform = 'translateY(-1px)'}
          onMouseLeave={e => e.target.style.transform = 'translateY(0)'}
        >
          classify
        </button>
      )}

      {/* PREDICTING STATE */}
      {ready && predicting && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
          padding: '14px 0', borderRadius: 'var(--radius-sm)',
          background: 'var(--surface)', border: '1px solid var(--border)'
        }}>
          <div style={{
            width: 16, height: 16, border: '2px solid var(--border)',
            borderTopColor: 'var(--accent)', borderRadius: '50%',
            animation: 'spin 0.8s linear infinite'
          }} />
          <span style={{ color: 'var(--ink-muted)', fontSize: 14 }}>classifying…</span>
        </div>
      )}

      {/* RESULT */}
      {result && (
        <div style={{
          borderRadius: 'var(--radius)', background: 'var(--surface)',
          border: '1px solid var(--border)', padding: 28, textAlign: 'center',
          animation: 'fade-up 0.5s var(--ease)',
          boxShadow: 'var(--shadow)'
        }}>
          <div style={{ fontSize: 52, lineHeight: 1, marginBottom: 10 }}>
            {result.label === 'DOG' ? '🐕' : '🐈'}
          </div>
          <div style={{
            fontFamily: "'DM Serif Display', serif",
            fontSize: 28, fontWeight: 400, letterSpacing: '-0.02em',
            color: result.label === 'DOG' ? 'var(--accent)' : 'var(--gold)',
            marginBottom: 16
          }}>
            {result.label.toLowerCase()}
          </div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            background: 'var(--bg)', borderRadius: 'var(--radius-sm)',
            padding: '10px 16px'
          }}>
            <span style={{ fontSize: 12, color: 'var(--ink-muted)', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              confidence
            </span>
            <div style={{
              flex: 1, height: 5, borderRadius: 3,
              background: 'var(--border)', overflow: 'hidden'
            }}>
              <div style={{
                width: `${result.confidence}%`, height: '100%',
                borderRadius: 3,
                background: result.label === 'DOG'
                  ? 'linear-gradient(90deg, var(--accent), #d48a5a)'
                  : 'linear-gradient(90deg, var(--gold), #c9a876)',
                transition: 'width 0.8s var(--ease)'
              }} />
            </div>
            <span style={{ color: 'var(--ink)', fontSize: 14, fontWeight: 600, minWidth: 40, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
              {result.confidence}%
            </span>
          </div>

          <button onClick={tryAgain}
            style={{
              marginTop: 18, padding: '9px 24px', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)', background: 'transparent',
              color: 'var(--ink-muted)', fontSize: 13, cursor: 'pointer',
              transition: `background var(--duration) var(--ease), border-color var(--duration) var(--ease)`
            }}
            onMouseEnter={e => { e.target.style.background = 'var(--surface-hover)'; e.target.style.borderColor = 'var(--ink-faint)' }}
            onMouseLeave={e => { e.target.style.background = 'transparent'; e.target.style.borderColor = 'var(--border)' }}
          >
            try another
          </button>
        </div>
      )}

      {/* PREDICTION ERROR */}
      {ready && errMsg && !image && (
        <div style={{
          padding: 16, borderRadius: 'var(--radius-sm)',
          background: '#fdf0ed', border: '1px solid #e8c8bd',
          textAlign: 'center', fontSize: 13, color: '#7a3e2e'
        }}>
          {errMsg}
        </div>
      )}

      {/* FOOTER */}
      <p style={{
        textAlign: 'center', color: 'var(--ink-faint)', fontSize: 11,
        letterSpacing: '0.02em', marginTop: 4
      }}>
        runs entirely in your browser &middot; no data uploaded
      </p>
    </div>
  )
}
