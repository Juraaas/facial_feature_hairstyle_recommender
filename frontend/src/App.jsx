import { useState } from 'react'
import { useAnalysis } from './hooks/useAnalysis'
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
        {/* upload */}
        <section className="upload-section">
          <div
            className="dropzone"
            onDrop={handleDrop}
            onDragOver={e => e.preventDefault()}
            onClick={() => document.getElementById('file-input').click()}
          >
            {preview
              ? <img src={preview} alt="uploaded" className="preview-img" />
              : <p>Drop a photo here or click to upload</p>
            }
          </div>
          <input
            id="file-input"
            type="file"
            accept="image/jpeg,image/png"
            onChange={handleUpload}
            style={{ display: 'none' }}
          />
          {file && !loading && (
            <button className="analyse-btn" onClick={handleAnalyse}>
              Analyse
            </button>
          )}
        </section>

        {/* loading */}
        {loading && (
          <div className="loading">
            <div className="spinner" />
            <p>Analysing photo...</p>
          </div>
        )}

        {/* error */}
        {error && (
          <div className="error-box">
            ⚠️ {error}
          </div>
        )}

        {/* results */}
        {result && (
          <section className="results">
            <p>Gender: {result.gender} | Confidence: {Math.round(result.quality.score * 100)}%</p>
            <p>Styles found: {result.styles.length}</p>
            <pre style={{fontSize: '11px', maxHeight: '200px', overflow: 'auto'}}>
              {JSON.stringify(result.traits, null, 2)}
            </pre>
          </section>
        )}
      </main>
    </div>
  )
}

export default App