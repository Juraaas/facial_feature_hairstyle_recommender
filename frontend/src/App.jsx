import { useEffect, useState } from 'react'
import { useAnalysis } from './hooks/useAnalysis'
import { FaceAnalysis }    from './components/FaceAnalysis'
import { FaceProportions } from './components/FaceProportions'
import { StylesSection }   from './components/StylesSection'
import { FeedbackSection } from './components/FeedbackSection'
import './App.css'

function App() {
  const { result, loading, error, analyse, reset} = useAnalysis()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [overlayUrl, setOverlayUrl] = useState(null)
  const [dark, setDark] = useState(false)

  useEffect(() => {
    document.body.setAttribute('data-theme', dark ? 'dark' : 'light')
  }, [dark])

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

  async function handleAnalyse() {
    if (!file) return
    
    const form = new FormData()
    form.append('file', file)
    const overlayPromise = fetch(`${import.meta.env.VITE_API_URL}/landmarks-overlay`, {
      method: 'POST', body: form
    }).then(r => r.blob()).then(b => URL.createObjectURL(b))
    
    analyse(file)
    setOverlayUrl(await overlayPromise)
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1>FaceFit AI</h1>
          <p>AI-powered hairstyle recommendations based on your face proportions 
            and key facial features</p>
        </div>
        <button
          className="theme-btn"
          onClick={() => setDark(d => !d)}
          title="Toggle dark mode"
        >
          {dark ? '☀️' : '🌙'}
        </button>
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
              : <p className="dropzone-hint">Drop a photo here or click to upload</p>
            }
          </div>
          <input id="file-input" type="file" accept="image/jpeg,image/png"
            onChange={e => handleFile(e.target.files[0])}
            style={{ display: 'none' }} />
          {file && !loading && !result && (
            <button className="analyse-btn" onClick={handleAnalyse}>
              Analyse
            </button>
          )}
          {result && (
            <button className="analyse-btn secondary"
              onClick={() => { reset(); setFile(null); setPreview(null); setOverlayUrl(null) }}>
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
            <div className="detection-bar">
              <span className="detection-gender">
                {result.gender === 'Woman' ? '👩' : '👨'} {result.gender} detected
              </span>
              <div className="confidence-track">
                <div className="confidence-fill" style={{
                  width: `${result.quality.score * 100}%`,
                  background: result.quality.score > 0.7 ? '#2d8f4e'
                            : result.quality.score > 0.4 ? '#e6a817' : '#c0392b'
                }} />
              </div>
              <span className="confidence-label">
                {Math.round(result.quality.score * 100)}%
              </span>
            </div>

            {result.quality.warnings?.map((w, i) => (
              <div key={i} className="warning-box">⚠️ {w}</div>
            ))}

            {/* visualization */}
            <section style={{ marginBottom: 32 }}>
              <h2 className="section-title">Visualization</h2>
              <div className="vis-grid">
                <div className="vis-item">
                  <img src={preview} alt="Original" className="vis-img" />
                  <p className="vis-label">Original</p>
                </div>
                <div className="vis-item">
                  {overlayUrl
                    ? <img src={overlayUrl} alt="Landmarks" className="vis-img" />
                    : <div className="vis-placeholder">Loading overlay...</div>
                  }
                  <p className="vis-label">Detected landmarks</p>
                </div>
              </div>
            </section>

            <FaceAnalysis analysis={result.analysis} />
            <FaceProportions features={result.features} norms={result.norms} />
            <StylesSection
              styles={result.styles}
              features={result.features}
              gender={result.gender}
            />
            <FeedbackSection
              features={result.features}
              qualityScore={result.quality.score}
              topStyles={result.styles.slice(0, 3)}
            />
          </>
        )}
      </main>
    </div>
  )
}

export default App