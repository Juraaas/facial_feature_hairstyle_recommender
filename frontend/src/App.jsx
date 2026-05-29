import { useState } from 'react'
import { useAnalysis } from './hooks/useAnalysis'
import { FaceAnalysis }    from './components/FaceAnalysis'
import { FaceProportions } from './components/FaceProportions'
import { StylesSection }   from './components/StylesSection'
import './App.css'

function App() {
  const { result, loading, error, analyse, reset} = useAnalysis()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)

  function handleFile(f) {
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    reset()
  }

  function handleUpload(e) {
    handleFile(e.target.files[0])
  }

  function handleDrop(e) {
    e.preventDefault()
    handleFile(e.dataTransfer.files[0])
  }

  function handleAnalyse() {
    if (file) analyse(file)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Hairstyle AI Recommender</h1>
      </header>

      <main className="app-main">
        <section style={{ marginBottom: 32 }}>
          <div
            className="dropzone"
            onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}
            onDragOver={e => e.preventDefault()}
            onClick={() => document.getElementById('file-input').click()}
          >
            {preview
              ? <img src={preview} alt="uploaded" className="preview-img" />
              : <p style={{ color: '#888' }}>Drop a photo here or click to upload</p>
            }
          </div>
          <input id="file-input" type="file" accept="image/jpeg,image/png"
            onChange={e => handleFile(e.target.files[0])}
            style={{ display: 'none' }} />
          {file && !loading && !result && (
            <button className="analyse-btn" onClick={() => analyse(file)}>
              Analyse
            </button>
          )}
          {result && (
            <button className="analyse-btn" style={{ background: '#888' }}
              onClick={() => { reset(); setFile(null); setPreview(null) }}>
              Upload new photo
            </button>
          )}
        </section>

        {loading && (
          <div className="loading">
            <div className="spinner" />
            <p>Analysing photo...</p>
          </div>
        )}

        {error && <div className="error-box">⚠️ {error}</div>}

        {result && (
          <>
            {/* detection bar */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 12,
              marginBottom: 24, padding: '8px 12px',
              background: '#fff', borderRadius: 8, border: '0.5px solid #e0e0e0'
            }}>
              <span style={{ fontSize: 13 }}>
                {result.gender === 'Woman' ? '👩' : '👨'} {result.gender} detected
              </span>
              <div style={{ flex: 1, height: 4, borderRadius: 2, background: '#eee' }}>
                <div style={{
                  width: `${result.quality.score * 100}%`, height: '100%',
                  borderRadius: 2,
                  background: result.quality.score > 0.7 ? '#2d8f4e'
                            : result.quality.score > 0.4 ? '#e6a817' : '#c0392b'
                }} />
              </div>
              <span style={{ fontSize: 12, fontWeight: 500 }}>
                {Math.round(result.quality.score * 100)}%
              </span>
            </div>

            {result.quality.warnings?.map((w, i) => (
              <div key={i} style={{
                background: '#fffbe6', border: '0.5px solid #ffe58f',
                borderRadius: 8, padding: '8px 12px', marginBottom: 8,
                fontSize: 13, color: '#856404'
              }}>⚠️ {w}</div>
            ))}

            <FaceAnalysis    analysis={result.analysis} />
            <FaceProportions features={result.features} norms={result.norms} />
            <StylesSection
              styles={result.styles}
              features={result.features}
              gender={result.gender}
            />
          </>
        )}
      </main>
    </div>
  )
}

export default App