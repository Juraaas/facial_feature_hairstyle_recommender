import { useState, useEffect } from 'react'
import { useAnalysis } from './hooks/useAnalysis'
import { FaceAnalysis } from './components/FaceAnalysis'
import { FaceProportions } from './components/FaceProportions'
import { StylesSection } from './components/StylesSection'
import { FeedbackSection } from './components/FeedbackSection'
import { ErrorBox } from './components/ErrorBox'
import { PhotoTutorial } from './components/PhotoTutorial'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { useAuth } from './hooks/useAuth'
import { AuthModal } from './components/AuthModal'
import { PremiumGate } from './components/PremiumGate'
import { PremiumPopup } from './components/PremiumPopup'
import { supabase } from './lib/supabase'
import { useDarkMode } from './hooks/useDarkMode'
import { useLocation } from 'react-router-dom'
import { btnOutline, darkToggleBtn, darkToggleTrack, darkToggleKnob } from './styles/shared'
import { StylePlayground } from './components/StylePlayground'
import { createCheckout } from './api/client'
import './App.css'

function App() {
  const { result, loading, error, analyse, reset} = useAnalysis()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dark, setDark] = useDarkMode()
  const [tutorialDone, setTutorialDone] = useState(
  () => localStorage.getItem('tutorial_done') === '1'
  )
  const [showTutorial, setShowTutorial] = useState(false)
  const { t, i18n } = useTranslation()
  const navigate = useNavigate()
  const pl = i18n.language === 'pl'

  const analysis = result?.face_analysis?.[i18n.language] || result?.face_analysis?.en || []
  const styles = result?.styles?.[i18n.language] || result?.styles?.en || []
  const { user, loading: authLoading, signOut, getToken } = useAuth()
  const [showAuth, setShowAuth] = useState(false)
  const [userPlan, setUserPlan] = useState('free')
  const isPremium = userPlan === 'premium'
  const [showPremium, setShowPremium] = useState(false)
  const location = useLocation()
  const [showPlayground, setShowPlayground] = useState(false)

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
    const token = await getToken()
    analyse(file, i18n.language, token)
  }

  async function handleUpgrade() {
    if (!user) { setShowAuth(true); return }
    try {
      await createCheckout()
    } catch (e) {
      console.error('Checkout error:', e)
    }
  }

  useEffect(() => {
    if (!user) { setUserPlan('free'); return }
    console.log('Fetching plan for user:', user.id)
    supabase
      .from('profiles')
      .select('plan')
      .eq('id', user.id)
      .single()
      .then(({ data, error }) => {
        console.log('Plan data:', data, error)
        if (data?.plan) setUserPlan(data.plan)
      })
  }, [user])

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)' }}>
      <header className="site-nav" style={{
          display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', padding: '16px 28px',  marginBottom: 28,
          borderBottom: '1px solid var(--border)', position: 'sticky',
          top: 0, background: 'var(--bg)', zIndex: 100}}>
          {/* logo + name nav home */}
          <div onClick={() => navigate('/')} style={{
              fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 500,
              letterSpacing: '.01em', cursor: 'pointer', display: 'flex',
              alignItems: 'center', gap: 8, color: 'var(--text)'}}>
            <img src="/android-chrome-192x192.png" alt=""
              style={{ width: 22, height: 22, borderRadius: 4 }} />
            Stylizzer
          </div>
          {/* right - settings + user */}
          <div className="header-right" style={{display: 'flex', gap: 6, alignItems: 'center'}}>
            <button onClick={() => setDark(d => !d)}
                style={darkToggleBtn(dark)}>
                <span style={{ fontSize: 12 }}>{dark ? '☀️' : '🌙'}</span>
                {/* track */}
                <span style={darkToggleTrack(dark)}>
                  {/* knob */}
                  <span style={darkToggleKnob(dark)} />
                </span>
              </button>
              <button onClick={toggleLang} style={btnOutline}>
                {i18n.language === 'pl' ? 'EN' : 'PL'}
              </button>
              {tutorialDone && !showTutorial && (
                <button onClick={() => setShowTutorial(true)} style={btnOutline}>
                  {t('photo_tips')}
                </button>
              )}
              <div style={{ width: 1, height: 16, background: 'var(--border)' }} />
              {user ? (
                <>
                  <span style={{
                    fontSize: 11, color: 'var(--text-muted)',
                    fontFamily: 'var(--font-mono)', padding: '5px 4px',
                  }}>
                    {user.email?.split('@')[0]}
                  </span>
                  <button onClick={signOut} style={btnOutline}>
                    {pl ? 'Wyloguj' : 'Sign out'}
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setShowAuth(true)}
                  style={{...btnOutline, borderColor: 'var(--accent)', color: 'var(--accent)'}}
                >
                  {pl ? 'Zaloguj' : 'Sign in'}
              </button>
              )}
              {showAuth && (
                <AuthModal
                  onClose={() => setShowAuth(false)}
                  onSuccess={() => setShowAuth(false)}
                />
              )}
          </div>
        </header>
      <div className="app">
        <main className="app-main">
          {/* intro */}
          {!result && !loading && (
            <div style={{ textAlign: 'center', padding: '32px 0 20px' }}>
              <h1 style={{
                fontFamily: 'var(--font-display)', fontSize: 'clamp(22px, 4vw, 32px)',
                fontWeight: 500, color: 'var(--text)', marginBottom: 8}}>
                {pl ? 'Twoja analiza twarzy' : 'Your face analysis'}
              </h1>
              <p style={{
                fontSize: 14, color: 'var(--text-muted)',
                fontWeight: 300, maxWidth: 480, margin: '0 auto'}}>
                {pl
                  ? 'Wrzuć zdjęcie i otrzymaj rekomendacje fryzur dopasowane do geometrii Twojej twarzy'
                  : 'Upload a photo and get hairstyle recommendations tailored to your face geometry'}
              </p>
            </div>
          )}

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
              {/* <PremiumGate isPremium={isPremium} onUnlock={() => setShowPremium(true)}> */}
                <FaceAnalysis analysis={analysis} />
              {/* </PremiumGate> */}
              <FaceProportions features={result.features} norms={result.norms} />
              {/* <PremiumGate isPremium={isPremium} onUnlock={() => setShowPremium(true)}> */}
                {result && styles.length > 0 && (
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
                    <button
                      onClick={() => setShowPlayground(true)}
                      style={{
                        ...btnOutline,borderColor: 'var(--accent)',
                        color: 'var(--accent)', gap: 6, display: 'flex', alignItems: 'center'
                      }}
                    >
                      ✨ {pl ? 'Przymierzalnia' : 'Style Playground'}
                    </button>
                  </div>
                )}
                <StylesSection
                  styles={styles}
                  features={result.features}
                  gender={result.gender}
                />
                {showPlayground && (
                  <StylePlayground
                    styles={styles}
                    originalFile={file}
                    isPremium={isPremium}
                    onUpgrade={() => setShowPlayground(false)}
                    onClose={() => setShowPlayground(false)}
                  />
                )}
              {/* </PremiumGate> */}

              {showPremium && (
                <PremiumPopup
                  onClose={() => setShowPremium(false)}
                  onUpgrade={() => {}}
                  onLogin={() => { setShowPremium(false); setShowAuth(true) }}
                  user={user}
                />
              )}
              <FeedbackSection
                features={result.features}
                qualityScore={result.quality.score}
                topStyles={styles.slice(0, 3)}
              />
            </>
          )}
        </main>
      </div>
    </div>
  )
}

export default App