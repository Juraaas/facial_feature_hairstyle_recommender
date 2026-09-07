import { useState, useEffect } from 'react'
import { useAnalysis } from './hooks/useAnalysis'
import { FaceAnalysis } from './components/FaceAnalysis'
import { FaceProportions } from './components/FaceProportions'
import { StylesSection } from './components/StylesSection'
import { FeedbackSection } from './components/FeedbackSection'
import { ErrorBox } from './components/ErrorBox'
import { PhotoTutorial } from './components/PhotoTutorial'
import { useTranslation } from 'react-i18next'
import { useAuth } from './hooks/useAuth'
import { PremiumGate } from './components/PremiumGate'
import { PremiumPopup } from './components/PremiumPopup'
import { supabase } from './lib/supabase'
import { useDarkMode } from './hooks/useDarkMode'
import { StylePlayground } from './components/StylePlayground'
import { UserPanel } from './components/UserPanel'
import { createCheckout } from './api/client'
import { NavBar } from './components/NavBar'
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
  const pl = i18n.language === 'pl'

  const analysis = result?.face_analysis?.[i18n.language] || result?.face_analysis?.en || []
  const styles = result?.styles?.[i18n.language] || result?.styles?.en || []
  const { user, loading: authLoading, signOut, getToken } = useAuth()
  const [userPlan, setUserPlan] = useState('free')
  const isPremium = userPlan === 'premium'
  const [showPremium, setShowPremium] = useState(false)
  const [showPlayground, setShowPlayground] = useState(false)
  const [showPanel, setShowPanel] = useState(false)

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
    supabase.auth.getSession().then(({ data: { session } }) => {
      console.log('Session exists:', !!session)
      
      supabase
        .from('profiles')
        .select('plan')
        .eq('id', user.id)
        .single()
        .then(({ data, error }) => {
          console.log('Plan data:', data, error)
          if (data?.plan) setUserPlan(data.plan)
        })
    })
  }, [user])

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)' }}
      data-theme={dark ? 'dark' : 'light'}>
      <NavBar
        variant="app"
        onShowPanel={() => setShowPanel(true)}
        onShowTutorial={() => setShowTutorial(true)}
        onUpgrade={handleUpgrade}
        isPremium={isPremium}
      />
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
              <PremiumGate isPremium={isPremium} onUnlock={() => setShowPremium(true)}>
                <FaceAnalysis analysis={analysis} />
              </PremiumGate>

              <FaceProportions features={result.features} norms={result.norms} />

              <StylesSection
                styles={isPremium ? styles : styles.slice(0, 3)}
                features={result.features}
                gender={result.gender}
                isPremium={isPremium}
                onPremiumClick={() => setShowPremium(true)}
              />

              {/* try-on */}
              <section style={{ marginBottom: 32 }}>
                <div style={{ marginBottom: 16 }}>
                  <div>
                    <h2 className="section-title" style={{ marginBottom: 4 }}>
                      {pl ? 'Przymierzalnia' : 'Style Playground'}
                    </h2>
                    <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 300 }}>
                      {pl ? 'Podgląd fryzury na Twoim zdjęciu' : 'Preview hairstyles on your photo'}
                    </p>
                  </div>
                </div>

                {!isPremium ? (
                  <div style={{
                    background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
                    border: '1px solid var(--border)', padding: '32px 24px', textAlign: 'center',
                  }}>
                    <div style={{
                      width: 48, height: 48, borderRadius: '50%',
                      background: 'var(--accent-soft)', border: '1.5px solid var(--accent)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 20, margin: '0 auto 16px',
                    }}>✨</div>
                    <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 15,
                      fontWeight: 500, color: 'var(--text)', marginBottom: 8 }}>
                      {pl ? 'Funkcja Premium' : 'Premium Feature'}
                    </h3>
                    <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300,
                      lineHeight: 1.6, maxWidth: 360, margin: '0 auto 20px' }}>
                      {pl
                        ? 'Sprawdź różne style na swoim zdjęciu.'
                        : 'Try on hairstyles and colors on your photo.'}
                    </p>
                    <button
                      onClick={() => setShowPremium(true)}
                      className="analyse-btn"
                      style={{ maxWidth: 240, margin: '0 auto' }}
                    >
                      {user ? (pl ? 'Kup Premium →' : 'Get Premium →')
                            : (pl ? 'Zaloguj się →' : 'Sign in →')}
                    </button>
                  </div>
                ) : (
                  <div
                    onClick={() => setShowPlayground(true)}
                    style={{
                      background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
                      border: '1px solid var(--border)', padding: '24px',
                      cursor: 'pointer', display: 'flex', alignItems: 'center',
                      gap: 16, transition: 'border-color .15s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--accent)'}
                    onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
                  >
                    <div style={{
                      width: 60, height: 60, borderRadius: 'var(--radius-md)',
                      background: 'var(--surface-2)', display: 'flex',
                      alignItems: 'center', justifyContent: 'center', fontSize: 24, flexShrink: 0,
                    }}>✂️</div>
                    <div>
                      <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--text)', marginBottom: 4 }}>
                        {pl ? 'Otwórz przymierzalnię' : 'Open Style Playground'}
                      </p>
                      <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 300 }}>
                        {pl ? 'Wybierz fryzurę i kolor'
                            : 'Choose a style and color'}
                      </p>
                    </div>
                    <span style={{ marginLeft: 'auto', color: 'var(--text-hint)', fontSize: 18 }}>→</span>
                  </div>
                )}
              </section>


              {showPlayground && (
                <StylePlayground
                  styles={styles}
                  originalFile={file}
                  isPremium={isPremium}
                  onUpgrade={() => {setShowPlayground(false); setShowPremium(true) }}
                  onClose={() => setShowPlayground(false)}
                />
              )}

              <FeedbackSection
                features={result.features}
                qualityScore={result.quality.score}
                topStyles={styles.slice(0, 3)}
              />

              {showPremium && (
                <PremiumPopup
                  onClose={() => setShowPremium(false)}
                  onUpgrade={handleUpgrade}
                  onLogin={() => { setShowPremium(false); setShowAuth(true) }}
                  user={user}
                />
              )}


            </>
          )}
        </main>
      </div>
      {showPanel && (
        <UserPanel
          user={user}
          isPremium={isPremium}
          onClose={() => setShowPanel(false)}
          onUpgrade={handleUpgrade}
        />
      )}
    </div>
  )
}

export default App