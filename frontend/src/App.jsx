import { useEffect, useState, useSyncExternalStore } from 'react'
import { useAnalysis } from './hooks/useAnalysis'
import { FaceAnalysis } from './components/FaceAnalysis'
import { FaceProportions } from './components/FaceProportions'
import { StylesSection } from './components/StylesSection'
import { FeedbackSection } from './components/FeedbackSection'
import { ErrorBox } from './components/ErrorBox'
import { PhotoTutorial } from './components/PhotoTutorial'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { supabase } from './lib/supabase'
import './App.css'

function App() {
  const { result, loading, error, analyse, reset} = useAnalysis()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dark, setDark] = useState(false)
  const [tutorialDone, setTutorialDone] = useState(
  () => localStorage.getItem('tutorial_done') === '1'
  )
  const [showTutorial, setShowTutorial] = useState(false)
  const { t, i18n } = useTranslation()
  const navigate = useNavigate()
  const pl = i18n.language === 'pl'

  const analysis = result?.face_analysis?.[i18n.language] || result?.face_analysis?.en || []
  const styles = result?.styles?.[i18n.language] || result?.styles?.en || []

  useEffect(() => {
    supabase.auth.getSession().then(({ data, error }) => {
      console.log('=== SUPABASE TEST ===')
      console.log('Session:', data.session)
      console.log('Error:', error)
    })
  }, [])

  useEffect(() => {
    document.body.setAttribute('data-theme', dark ? 'dark' : 'light')
  }, [dark])

  function handleFile(f) {
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    reset()
  }

  function handleTutorialDone() {
    localStorage.setItem('tutorial_done', '1')
    setTutorialDone(true)
    setShowTutorial(false)
  }

  function toggleLang() {
    const next = i18n.language === 'pl' ? 'en' : 'pl'
    i18n.changeLanguage(next)
    localStorage.setItem('lang', next)
  }

  async function handleAnalyse() {
    if (!file) return
    analyse(file, i18n.language)
  }

  const headerBtnStyle = {
    background: 'none',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
    fontSize: 11,
    color: 'var(--text-muted)',
    padding: '5px 10px',
    fontFamily: 'var(--font-body)',
    letterSpacing: '.02em',
    transition: 'border-color .15s, color .15s',
    whiteSpace: 'nowrap',
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1>{t('app_title')}</h1>
          <p>{t('app_subtitle')}</p>
        </div>
        <div style={{ position: 'absolute', right: 0, top: 0, display: 'flex', 
          flexDirection: 'row', gap: 6, alignItems: 'flex-end',
         }}>
          <button onClick={() => setDark(d => !d)}
            style={{
              ...headerBtnStyle, display: 'flex', alignItems: 'center', gap: 5, padding: '5px 8px'}}>
            {/* track */}
            <span style={{
              width: 28, height: 14, borderRadius: 7, background: dark ? 'var(--accent)' : 'var(--border)',
              position: 'relative', display: 'block', transition: 'background .2s', flexShrink: 0,
            }}>
              {/* knob */}
              <span style={{
                position: 'absolute', top: 2, left: dark ? 14 : 2, width: 10, height: 10,
                borderRadius: '50%', background: '#fff', transition: 'left .2s'
              }} />
            </span>
          </button>
          <button onClick={toggleLang} style={headerBtnStyle}>
            {i18n.language === 'pl' ? 'EN' : 'PL'}
          </button>
          {tutorialDone && !showTutorial && (
            <button onClick={() => setShowTutorial(true)} style={headerBtnStyle}>
              {t('photo_tips')}
            </button>
          )}
          <button onClick={() => navigate('/')} style={headerBtnStyle}>
            ← {pl ? 'Strona główna' : 'Home'}
          </button>
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
                onClick={() => { reset(); setFile(null); setPreview(null); }}>
                {t('btn_upload_new')}
              </button>
            )}
          </section>
        )}

        {loading && (
          <div className="loading">
            <div className="spinner" />
            <p style={{ fontSize: 13, fontWeight: 300 }}>{t('btn_loading')}</p>
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
                            : result.quality.score > 0.4 ? '#C8975A' : '#c0392b'
                }} />
              </div>
              <span className="confidence-label">
                {Math.round(result.quality.score * 100)}%
              </span>
            </div>

            {result.quality.warnings?.map((w, i) => (
              <div key={i} className="warning-box">⚠️ {w}</div>
            ))}

            {/* hair trait badges */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
              {[
                {
                  icon:  '💇',
                  label: result.traits?.hair_type
                    ? `${t(`hair_type_${result.traits.hair_type}`)} ${t('hair_type_label')}`
                    : t('hair_type_not_detected'),
                  dashed: !result.traits?.hair_type,
                },
                {
                  icon:  '📐',
                  label: (result.traits?.hairline && result.traits.hairline !== 'normal')
                    ? t(`hairline_${result.traits.hairline}`)
                    : t('hairline_normal'),
                  dashed: false,
                },
              ].map(({ icon, label, dashed }) => (
                <div key={label} style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: 11,
                  padding: '4px 12px',
                  borderRadius: 20,
                  background: 'var(--surface)',
                  border: `1px ${dashed ? 'dashed' : 'solid'} var(--border)`,
                  color: 'var(--text-muted)',
                  fontWeight: 300,
                }}>
                  <span>{icon}</span>
                  <span>{label}</span>
                </div>
              ))}
            </div>
            <FaceAnalysis analysis={analysis} />
            <FaceProportions features={result.features} norms={result.norms} />
            <StylesSection
              styles={styles}
              features={result.features}
              gender={result.gender}
            />
            <FeedbackSection
              features={result.features}
              qualityScore={result.quality.score}
              topStyles={styles.slice(0, 3)}
            />
          </>
        )}
      </main>
    </div>
  )
}

export default App