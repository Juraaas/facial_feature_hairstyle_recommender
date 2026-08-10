import { useEffect, useState, useSyncExternalStore } from 'react'
import { useAnalysis } from './hooks/useAnalysis'
import { FaceAnalysis } from './components/FaceAnalysis'
import { FaceProportions } from './components/FaceProportions'
import { StylesSection } from './components/StylesSection'
import { FeedbackSection } from './components/FeedbackSection'
import { ErrorBox } from './components/ErrorBox'
import { PhotoTutorial } from './components/PhotoTutorial'
import { useTranslation } from 'react-i18next'
import './App.css'

function App() {
  const { result, loading, error, analyse, reset} = useAnalysis()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [overlayUrl, setOverlayUrl] = useState(null)
  const [dark, setDark] = useState(false)
  const [tutorialDone, setTutorialDone] = useState(
  () => localStorage.getItem('tutorial_done') === '1'
  )
  const [showTutorial, setShowTutorial] = useState(false)
  const { t, i18n } = useTranslation()

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

  function handleTutorialDone() {
    localStorage.setItem('tutorial_done', '1')
    setTutorialDone(true)
    setShowTutorial(false)
  }

  function toggleLang() {
    const next = i18n.language === 'pl' ? 'en' : 'pl'
    console.log('changing language:', i18n.language, '→', next)
    i18n.changeLanguage(next)
    localStorage.setItem('lang', next)
  }

  async function handleAnalyse() {
    if (!file) return
    
    const form = new FormData()
    const lang = i18n.language
    form.append('file', file)
    const overlayPromise = fetch(`${import.meta.env.VITE_API_URL}/landmarks-overlay`, {
      method: 'POST', body: form
    }).then(r => r.blob()).then(b => URL.createObjectURL(b))
    
    analyse(file, lang)
    setOverlayUrl(await overlayPromise)
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1>{t('app_title')}</h1>
          <p>{t('app_subtitle')}</p>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-end' }}>
          <button className="theme-btn" onClick={() => setDark(d => !d)}>
            {dark ? '☀️' : '🌙'}
          </button>
          <button
            onClick={toggleLang}
            style={{
              background: 'none',
              border: '0.5px solid var(--border)',
              borderRadius: 6, cursor: 'pointer',
              fontSize: 11, color: 'var(--text-muted)',
              padding: '4px 8px',
            }}
          >
            {i18n.language === 'pl' ? '🇬🇧 EN' : '🇵🇱 PL'}
          </button>
          {tutorialDone && !showTutorial && (
            <button onClick={() => setShowTutorial(true)}
            style={{
              background: 'none', border: '0.5px solid var(--border)', borderRadius: 6,
              cursor: 'pointer', fontSize: 11, color: 'var(--text-muted)', padding: '4px 8px',
            }}>
              {t('photo_tips')}
            </button>
          )}
        </div>
      </header>

      <main className="app-main">
        {(showTutorial || !tutorialDone) ? (
          <PhotoTutorial onDone={handleTutorialDone} />
        ) : (
          <section style={{ marginBottom: 32 }}>
            <div
              className="dropzone"
              onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}
              onDragOver={e => e.preventDefault()}
              onClick={() => document.getElementById('file-input').click()}
            >
              {preview
                ? <img src={preview} alt="uploaded" className="preview-img" />
                : <p className="dropzone-hint">{t('dropzone_hint')}</p>
              }
            </div>
            <input id="file-input" type="file" accept="image/jpeg,image/png"
              onChange={e => handleFile(e.target.files[0])}
              style={{ display: 'none' }} />
            {file && !loading && !result && (
              <button className="analyse-btn" onClick={handleAnalyse}>
                {t('btn_analyse')}
              </button>
            )}
            {result && (
              <button className="analyse-btn secondary"
                onClick={() => { reset(); setFile(null); setPreview(null); setOverlayUrl(null) }}>
                {t('btn_upload_new')}
              </button>
            )}
          </section>
        )}

        {loading && (
          <div className="loading">
            <div className="spinner" />
            <p>{t('btn_loading')}</p>
          </div>
        )}

        {error && <ErrorBox error={error} />}

        {result && (
          <>
            <div className="detection-bar">
              <span className="detection-gender">
                {result.gender === 'Woman' ? t('detected_woman'): t('detected_man')}
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

            {/* hair traits badge */}
            <div style={{
              display: 'flex', gap: 8, flexWrap: 'wrap',
              marginBottom: 16
            }}>
              {result.traits?.hair_type ? (
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  fontSize: 12, padding: '4px 10px', borderRadius: 20,
                  background: 'var(--surface)', border: '0.5px solid var(--border)',
                  color: 'var(--text-muted)'
                }}>
                  <span>💇</span>
                  <span>{result.traits.hair_type} {t('hair_type_label')}</span>
                </div>
              ) : (
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  fontSize: 12, padding: '4px 10px', borderRadius: 20,
                  background: 'var(--surface)', border: '0.5px dashed var(--border)',
                  color: 'var(--text-muted)'
                }}>
                  <span>💇</span>
                  <span>{t('hair_type_not_detected')}</span>
                </div>
              )}

              {result.traits?.hairline && result.traits.hairline !== 'normal' ? (
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  fontSize: 12, padding: '4px 10px', borderRadius: 20,
                  background: 'var(--surface)', border: '0.5px solid var(--border)',
                  color: 'var(--text-muted)'
                }}>
                  <span>📐</span>
                  <span>{result.traits.hairline} {t('hairline_label')}</span>
                </div>
              ) : (
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  fontSize: 12, padding: '4px 10px', borderRadius: 20,
                  background: 'var(--surface)', border: '0.5px solid var(--border)',
                  color: 'var(--text-muted)'
                }}>
                  <span>📐</span>
                  <span>{t('hairline_normal')}</span>
                </div>
              )}
            </div>

            {/* visualization */}
            <section style={{ marginBottom: 32 }}>
              <h2 className="section-title">{t('section_visualization')}</h2>
              <div className="vis-grid">
                <div className="vis-item">
                  <img src={preview} alt="Original" className="vis-img" />
                  <p className="vis-label">{t('label_original')}</p>
                </div>
                <div className="vis-item">
                  {overlayUrl
                    ? <img src={overlayUrl} alt="Landmarks" className="vis-img" />
                    : <div className="vis-placeholder">{t('load_ovelray')}</div>
                  }
                  <p className="vis-label">{t('label_landmarks')}</p>
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