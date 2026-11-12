import { useState, useEffect, useCallback } from 'react'
import { useAnalysis } from './hooks/useAnalysis'
import { FaceAnalysis } from './components/FaceAnalysis'
import { FaceProportions } from './components/FaceProportions'
import { StylesSection } from './components/StylesSection'
import { FeedbackSection } from './components/FeedbackSection'
import { ErrorBox } from './components/ErrorBox'
import { PhotoTutorial } from './components/PhotoTutorial'
import { useTranslation } from 'react-i18next'
import { useAuth } from './hooks/useAuth'
import { useDarkMode } from './hooks/useDarkMode'
import { StylePlayground } from './components/StylePlayground'
import { UserPanel } from './components/UserPanel'
import { AuthModal } from './components/AuthModal'
import { NavBar } from './components/NavBar'
import { ConsentCheckboxes } from './components/ConsentCheckboxes'
import { FadeLevel } from './components/FadeLevel'
import { HairColorSuggestions } from './components/HairColorSuggestions'
import { BuyCreditsModal } from './components/BuyCreditsModal'
import { UnlockCard } from './components/UnlockCard'
import { creditWord, plural } from './lib/credits'
import { fetchMe, createCheckout, API_BASE } from './api/client'
import { X, Scissors, ArrowRight, ScanFace, Waves, ScanLine, Brush, LogIn } from 'lucide-react'
import './App.css'

const FREE_FIX_MS = 14 * 60 * 1000

function App() {
  const { result, loading, warming, error, analyse, reset } = useAnalysis()
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dark] = useDarkMode()
  const [tutorialDone, setTutorialDone] = useState(() => localStorage.getItem('tutorial_done') === '1')
  const [showTutorial, setShowTutorial] = useState(false)
  const [showPlayground, setShowPlayground] = useState(false)
  const [showPanel, setShowPanel] = useState(false)
  const [showAuth, setShowAuth] = useState(false)
  const [showBuy, setShowBuy] = useState(false)
  const [buying, setBuying] = useState(null)
  const [consent, setConsent] = useState(false)
  const [overrideGender, setOverrideGender] = useState(null)
  const [fixInfo, setFixInfo] = useState({ at: 0, n: 0 })
  const [me, setMe] = useState(null)
  const [notice, setNotice] = useState(null)
  const [paymentStatus] = useState(() => new URLSearchParams(window.location.search).get('payment'))
  const [paymentDone, setPaymentDone] = useState(false)
  const [meChecked, setMeChecked] = useState(false)

  const { t, i18n } = useTranslation()
  const pl = i18n.language === 'pl'
  const { user } = useAuth()

  const credits = me?.credits ?? null
  const costs = me?.costs ?? { analysis: 1, preview: 3 }
  const isFull = result?.access === 'full'
  const canFull = !!user && credits !== null && credits >= costs.analysis
  const costWord = creditWord(costs.analysis, pl)
  const meLoading = !!user && !meChecked

  const analysis = result?.face_analysis?.[i18n.language] || result?.face_analysis?.en || []
  const styles = result?.styles?.[i18n.language] || result?.styles?.en || []

  useEffect(() => { fetch(`${API_BASE}/health`, { mode: 'no-cors' }).catch(() => {}) }, [])

  useEffect(() => {
    if (paymentStatus) window.history.replaceState({}, '', window.location.pathname)
    if (paymentStatus === 'cancelled') {
      setNotice({ type: 'info', text: pl ? 'Płatność została anulowana.' : 'Payment was cancelled.' })
    }
  }, [])

  const uid = user?.id ?? null

  const refreshMe = useCallback(async () => {
    if (!uid) { setMe(null); setMeChecked(false); return null }
    try {
      const data = await fetchMe()
      setMe(data)
      return data
    } catch (e) {
      console.error('fetchMe failed', e)
      return null
    } finally {
      setMeChecked(true)
    }
  }, [uid])

  useEffect(() => { setMeChecked(false); refreshMe() }, [refreshMe])

  useEffect(() => {
    if (result?.credits != null) setMe(m => (m ? { ...m, credits: result.credits } : m))
  }, [result])

  useEffect(() => {
    if (paymentStatus !== 'success' || !uid || paymentDone) return
    const before = Number(sessionStorage.getItem('credits_before') ?? -1)
    setNotice({ type: 'info', text: pl ? 'Księgujemy płatność…' : 'Processing payment…' })
    let tries = 0
    const timer = setInterval(async () => {
      tries += 1
      const data = await refreshMe()
      if (data && data.credits > before) {
        clearInterval(timer); setPaymentDone(true)
        sessionStorage.removeItem('credits_before')
        setNotice({ type: 'ok', text: pl
          ? `Płatność zaksięgowana. Masz ${data.credits} ${creditWord(data.credits, true)}.`
          : `Payment received. You have ${data.credits} ${creditWord(data.credits, false)}.` })
      } else if (tries >= 20) {
        clearInterval(timer); setPaymentDone(true)
        setNotice({ type: 'info', text: pl
          ? 'Płatność jest jeszcze przetwarzana. Odśwież stronę za chwilę. Jeśli kredyty nie pojawią się w ciągu kilku minut, napisz do nas.'
          : 'Your payment is still processing. Refresh in a moment. If credits do not appear within a few minutes, contact us.' })
      }
    }, 2000)
    return () => clearInterval(timer)
  }, [paymentStatus, uid, paymentDone, refreshMe, pl])

  function openBuy() {
    if (!user) { setShowAuth(true); return }
    setShowBuy(true)
  }

  async function handleBuy(packId) {
    setBuying(packId)
    try {
      sessionStorage.setItem('credits_before', String(credits ?? 0))
      await createCheckout(packId)
    } catch {
      setBuying(null)
      setNotice({ type: 'error', text: pl
        ? 'Nie udało się rozpocząć płatności. Spróbuj ponownie.'
        : 'Could not start checkout. Please try again.' })
    }
  }

  function clearFile() {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null); setPreview(null); reset()
    setOverrideGender(null); setConsent(false)
    const input = document.getElementById('file-input')
    if (input) input.value = ''
  }

  function handleFile(f) {
    if (!f) return
    if (preview) URL.revokeObjectURL(preview)
    setFile(f); setPreview(URL.createObjectURL(f)); reset()
    setOverrideGender(null); setConsent(false)
  }

  function handleTutorialDone() {
    localStorage.setItem('tutorial_done', '1')
    setTutorialDone(true); setShowTutorial(false)
  }

  async function runAnalysis({ gender = null, full, analysisId = null } = {}) {
    if (!file) return null
    const wantFull = full ?? canFull
    return analyse(file, i18n.language, { gender: gender ?? overrideGender, full: wantFull, analysisId })
  }

  async function handleAnalyse() {
    const data = await runAnalysis()
    if (data?.access === 'full') setFixInfo({ at: Date.now(), n: 0 })
  }

  async function handleUnlock() {
    if (!user) { setShowAuth(true); return }
    if ((credits ?? 0) < costs.analysis) { openBuy(); return }
    const data = await runAnalysis({ full: true, gender: result?.gender })
    if (data?.access === 'full') setFixInfo({ at: Date.now(), n: 0 })
  }

  async function handleGenderFix() {
    const corrected = result.gender === 'Woman' ? 'Man' : 'Woman'
    setOverrideGender(corrected)
    if (!isFull) { await runAnalysis({ gender: corrected, full: false }); return }

    const free = fixInfo.n < 2 && Date.now() - fixInfo.at < FREE_FIX_MS
    if (!free) {
      if ((credits ?? 0) < costs.analysis) { openBuy(); return }
      const ok = window.confirm(pl
        ? `Darmowa korekta wygasła. Nowa analiza pobierze ${costs.analysis} ${costWord}. Kontynuować?`
        : `The free correction has expired. A new analysis costs ${costs.analysis} ${costWord}. Continue?`)
      if (!ok) return
    }
    const data = await runAnalysis({ gender: corrected, full: true, analysisId: result.analysis_id })
    if (data?.access === 'full') {
      setFixInfo(free ? { at: fixInfo.at, n: fixInfo.n + 1 } : { at: Date.now(), n: 0 })
    }
  }

  const analyseSuffix = canFull
    ? ` · ${costs.analysis} ${costWord}`
    : ` · ${pl ? 'wersja podstawowa' : 'basic version'}`
  const analyseHint = !user
    ? (pl ? 'Zaloguj się, aby dostać pierwszą pełną analizę za darmo.' : 'Sign in to get your first full analysis for free.')
    : !canFull
      ? (pl ? `Pełna analiza kosztuje ${costs.analysis} ${costWord}. Teraz zobaczysz wersję podstawową.`
            : `A full analysis costs ${costs.analysis} ${costWord}. You will get the basic version now.`)
      : ''

  const lockedStyles = result?.locked_styles ?? 0
  const lacking = !!user && (credits ?? 0) < costs.analysis
  const benefits = pl
    ? [
        'opis dopasowania',
        lockedStyles > 0 ? `jeszcze ${lockedStyles} ${plural(lockedStyles, 'fryzurę', 'fryzury', 'fryzur')} z uzasadnieniami` : null,
        result?.gender === 'Man' ? 'poziom fade' : null,
        'pełną paletę kolorów',
      ].filter(Boolean).join(', ')
    : [
        'a personal fit summary',
        lockedStyles > 0 ? `${lockedStyles} more ${lockedStyles === 1 ? 'hairstyle' : 'hairstyles'} with explanations` : null,
        result?.gender === 'Man' ? 'your fade level' : null,
        'the full color palette',
      ].filter(Boolean).join(', ')

  const unlockText = !user
    ? (pl ? `Zaloguj się i odblokuj: ${benefits}. Pierwsza analiza jest za darmo.`
          : `Sign in to unlock: ${benefits}. Your first analysis is free.`)
    : lacking
      ? (pl ? `Pełna analiza kosztuje ${costs.analysis} ${costWord}. Kup kredyty, aby odblokować: ${benefits}.`
            : `A full analysis costs ${costs.analysis} ${costWord}. Buy credits to unlock: ${benefits}.`)
      : (pl ? `Odblokuj za ${costs.analysis} ${costWord}: ${benefits}.`
            : `Unlock for ${costs.analysis} ${costWord}: ${benefits}.`)
  const unlockCta = !user ? (pl ? 'Zaloguj się' : 'Sign in')
    : lacking ? (pl ? 'Kup kredyty' : 'Buy credits')
    : (pl ? 'Odblokuj' : 'Unlock')

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)' }}
      data-theme={dark ? 'dark' : 'light'}>
      <NavBar
        variant="app"
        onShowPanel={() => setShowPanel(true)}
        onShowTutorial={() => setShowTutorial(true)}
        onBuy={openBuy}
        credits={user ? credits : null}
      />
      <div className="app">
        <main className="app-main">
          {notice && (
            <div role="status" style={{
              display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px', marginBottom: 16,
              borderRadius: 'var(--radius-md)', fontSize: 13,
              border: `1px solid ${notice.type === 'ok' ? '#b7dfc7' : notice.type === 'error' ? '#f5c6bc' : 'var(--border)'}`,
              background: notice.type === 'ok' ? '#f0faf4' : notice.type === 'error' ? '#fef4f2' : 'var(--surface)',
              color: notice.type === 'ok' ? '#2d8f4e' : notice.type === 'error' ? '#c0392b' : 'var(--text-muted)',
            }}>
              <span style={{ flex: 1 }}>{notice.text}</span>
              <button onClick={() => setNotice(null)} aria-label="close" style={{
                background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', display: 'flex',
              }}><X size={14} /></button>
            </div>
          )}

          {/* intro */}
          {!result && !loading && tutorialDone && !showTutorial && (
            <div style={{ textAlign: 'center', padding: '32px 0 20px' }}>
              <h1 style={{
                fontFamily: 'var(--font-display)', fontSize: 'clamp(22px, 4vw, 32px)',
                fontWeight: 500, color: 'var(--text)', marginBottom: 8 }}>
                {pl ? 'Twoja analiza twarzy' : 'Your face analysis'}
              </h1>
              <p style={{ fontSize: 14, color: 'var(--text-muted)', fontWeight: 300,
                maxWidth: 480, margin: '0 auto' }}>
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
              <div style={{ position: 'relative' }}>
                <div
                  className="dropzone"
                  onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}
                  onDragOver={e => e.preventDefault()}
                  onClick={() => !preview && document.getElementById('file-input').click()}
                  style={{ cursor: preview ? 'default' : 'pointer' }}
                >
                  {preview
                    ? <img src={preview} alt="uploaded" className="preview-img" />
                    : <p className="dropzone-hint">{t('dropzone_hint')}</p>}
                </div>
                {preview && (
                  <button
                    onClick={e => { e.stopPropagation(); clearFile() }}
                    style={{
                      position: 'absolute', top: 8, right: 8, width: 28, height: 28,
                      borderRadius: '50%', background: 'rgba(15,15,14,.7)',
                      backdropFilter: 'blur(4px)', border: 'none', cursor: 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: '#fff', zIndex: 10, transition: 'background .15s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(200,0,0,.8)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'rgba(15,15,14,.7)'}
                  >
                    <X size={14} strokeWidth={2} />
                  </button>
                )}
              </div>
              <input id="file-input" type="file" accept="image/jpeg,image/png"
                onChange={e => handleFile(e.target.files[0])} style={{ display: 'none' }} />
              {file && !loading && !result && (
                <>
                  <ConsentCheckboxes onChange={setConsent} />
                  <button className="analyse-btn" onClick={handleAnalyse} disabled={!consent || meLoading}
                    style={{ opacity: consent && !meLoading ? 1 : 0.45 }}>
                    {t('btn_analyse')}{analyseSuffix}
                  </button>
                  {analyseHint && (
                    <p style={{ fontSize: 11, color: 'var(--text-hint)', textAlign: 'center',
                      marginTop: 8, fontWeight: 300 }}>{analyseHint}</p>
                  )}
                </>
              )}
              {result && (
                <button className="analyse-btn secondary" onClick={clearFile}>
                  {t('btn_upload_new')}
                </button>
              )}
            </section>
          )}

          {loading && (
            <div className="loading">
              <div className="spinner" />
              <p style={{ fontSize: 13, fontWeight: 300 }}>
                {warming
                  ? (pl ? 'Serwer się uruchamia, to może potrwać do minuty…'
                        : 'The server is starting up, this can take up to a minute…')
                  : t('btn_loading')}
              </p>
            </div>
          )}

          {error && <ErrorBox error={error} />}

          {result && (
            <>
              <div style={{
                background: 'var(--surface)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)', marginBottom: 20, overflow: 'hidden',
              }}>
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '8px 14px', borderBottom: '1px solid var(--border)',
                }}>
                  <span style={{ fontSize: 14, fontWeight: 500, color: 'var(--text)',
                    whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: 5, flex: 1 }}>
                    <ScanFace size={14} strokeWidth={1.5} color="var(--accent)" />
                    {result.gender === 'Woman' ? t('detected_woman') : t('detected_man')}
                  </span>
                  <button
                    onClick={handleGenderFix}
                    title={pl ? 'Korekta płci jest bezpłatna przez 15 minut (2 razy)'
                              : 'Gender correction is free for 15 minutes (2 times)'}
                    style={{ fontSize: 10, padding: '3px 10px', borderRadius: 20,
                      border: '1px solid var(--border)', background: 'none', color: 'var(--text-hint)',
                      cursor: 'pointer', fontFamily: 'var(--font-mono)', letterSpacing: '.04em' }}
                  >
                    {result.gender === 'Woman'
                      ? (pl ? 'To mężczyzna' : 'Switch to male')
                      : (pl ? 'To kobieta' : 'Switch to female')}
                  </button>
                  <div className="confidence-track" style={{ flex: 1 }}>
                    <div className="confidence-fill" style={{
                      width: `${result.quality.score * 100}%`,
                      background: result.quality.score > 0.7 ? '#2d8f4e'
                                : result.quality.score > 0.4 ? '#C8975A' : '#c0392b',
                    }} />
                  </div>
                  <span className="confidence-label">{Math.round(result.quality.score * 100)}%</span>
                </div>

                <div style={{ display: 'flex', padding: '8px 14px', gap: 0 }}>
                  {[
                    {
                      Icon: Waves,
                      label: result.traits?.hair_type
                        ? `${t(`hair_type_${result.traits.hair_type}`)} ${t('hair_type_label')}`
                        : t('hair_type_not_detected'),
                      dashed: !result.traits?.hair_type,
                    },
                    {
                      Icon: ScanLine,
                      label: result.traits?.hairline
                        ? t(`hairline_${result.traits.hairline}`)
                        : t('hairline_not_detected'),
                      dashed: !result.traits?.hairline,
                    },
                  ].map(({ Icon, label, dashed }, idx) => {
                    const ItemIcon = Icon
                    return (
                      <div key={label} style={{
                        display: 'flex', alignItems: 'center', gap: 6, flex: 1,
                        padding: '2px 8px', borderLeft: idx > 0 ? '1px solid var(--border)' : 'none',
                      }}>
                        <ItemIcon size={14} strokeWidth={1.5} color="var(--accent)" />
                        <span style={{
                          fontSize: 14, color: dashed ? 'var(--text-hint)' : 'var(--text-muted)',
                          fontWeight: 300, fontStyle: dashed ? 'italic' : 'normal',
                          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                        }}>{label}</span>
                      </div>
                    )
                  })}
                </div>
              </div>

              {isFull && analysis.length > 0 && <FaceAnalysis analysis={analysis} />}

              <HairColorSuggestions recommendation={result.color_recommendation} />

              <FaceProportions features={result.features} norms={result.norms} traits={result.traits} />

              <StylesSection
                styles={styles}
                features={result.features}
                gender={result.gender}
                unlocked={isFull}
                onUnlock={handleUnlock}
              />

              {!isFull && (
                <UnlockCard
                  title={pl ? 'Pełna analiza' : 'Full analysis'}
                  text={unlockText}
                  cta={unlockCta}
                  onClick={handleUnlock}
                />
              )}

              {result.fade_recommendation && result.gender === 'Man' && (
                <FadeLevel recommendation={result.fade_recommendation} />
              )}

              <section style={{ marginBottom: 32 }}>
                <div style={{ marginBottom: 16 }}>
                  <h2 className="section-title" style={{ display: 'flex', alignItems: 'center',
                    gap: 8, marginBottom: 4 }}>
                    <Brush size={18} color="var(--accent)" strokeWidth={1.5} />
                    {pl ? 'Przymierzalnia' : 'Style Playground'}
                  </h2>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 300, textAlign: 'left' }}>
                    {pl ? 'Podgląd fryzury na Twoim zdjęciu' : 'Preview hairstyles on your photo'}
                  </p>
                </div>

                {!user ? (
                  <div style={{
                    background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
                    border: '1px solid var(--border)', padding: '28px 24px', textAlign: 'center',
                  }}>
                    <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300,
                      lineHeight: 1.6, maxWidth: 360, margin: '0 auto 18px' }}>
                      {pl ? 'Zaloguj się, aby sprawdzić fryzury i kolory na swoim zdjęciu.'
                          : 'Sign in to try hairstyles and colors on your photo.'}
                    </p>
                    <button className="analyse-btn" onClick={() => setShowAuth(true)}
                      style={{ maxWidth: 240, margin: '0 auto', display: 'flex',
                        alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                      <LogIn size={14} strokeWidth={1.5} />
                      {pl ? 'Zaloguj się' : 'Sign in'}
                    </button>
                  </div>
                ) : (
                  <div
                    onClick={() => setShowPlayground(true)}
                    style={{
                      background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
                      border: '1px solid var(--border)', padding: '24px', cursor: 'pointer',
                      display: 'flex', alignItems: 'center', gap: 16, transition: 'border-color .15s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--accent)'}
                    onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
                  >
                    <div style={{
                      width: 56, height: 56, borderRadius: 'var(--radius-md)',
                      background: 'var(--surface-2)', display: 'flex',
                      alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                    }}>
                      <Scissors size={24} color="var(--accent)" strokeWidth={1.5} />
                    </div>
                    <div>
                      <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--text)', marginBottom: 4 }}>
                        {pl ? 'Otwórz przymierzalnię' : 'Open Style Playground'}
                      </p>
                      <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 300 }}>
                        {pl ? `Podgląd kosztuje ${costs.preview} ${creditWord(costs.preview, true)}`
                            : `A preview costs ${costs.preview} ${creditWord(costs.preview, false)}`}
                      </p>
                    </div>
                    <ArrowRight size={16} color="var(--text-hint)" style={{ marginLeft: 'auto' }} />
                  </div>
                )}
              </section>

              <FeedbackSection
                features={result.features}
                qualityScore={result.quality.score}
                topStyles={styles.slice(0, 3)}
              />

              {showPlayground && (
                <StylePlayground
                  styles={styles}
                  originalFile={file}
                  analysisResult={result}
                  gender={result.gender}
                  credits={credits ?? 0}
                  previewCost={costs.preview}
                  onCreditsChange={c => setMe(m => (m ? { ...m, credits: c } : m))}
                  onNeedCredits={() => { setShowPlayground(false); openBuy() }}
                  onClose={() => setShowPlayground(false)}
                />
              )}
            </>
          )}
        </main>
      </div>

      {showBuy && (
        <BuyCreditsModal
          packs={me?.packs ?? []}
          costs={costs}
          credits={credits ?? 0}
          buying={buying}
          onBuy={handleBuy}
          onClose={() => setShowBuy(false)}
        />
      )}

      {showPanel && (
        <UserPanel
          user={user}
          credits={credits ?? 0}
          onClose={() => setShowPanel(false)}
          onBuy={() => { setShowPanel(false); openBuy() }}
        />
      )}

      {showAuth && (
        <AuthModal onClose={() => setShowAuth(false)} onSuccess={() => setShowAuth(false)} />
      )}
    </div>
  )
}

export default App