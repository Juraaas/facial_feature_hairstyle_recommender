import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL
const DEMO_CASES_MAN = [
  {
    face: 'Long face · Narrow jaw',
    face_pl: 'Długa twarz · Wąska szczęka',
    analysis: 'Side volume and fringe optically widen your face and shorten its length.',
    analysis_pl: 'Objętość po bokach i grzywka optycznie poszerzają twarz i skracają ją.',
    styles: ['French Crop', 'Textured Fringe'],
    image: `${API_URL}/images/male/french_crop.jpg`,
  },
  {
    face: 'Wide jaw · Short face',
    face_pl: 'Szeroka szczęka · Szeroka twarz',
    analysis: 'Height on top and tapered sides balance a broader jaw and create a more elongated silhouette.',
    analysis_pl: 'Objętość na górze i krótkie boki równoważą szeroką szczękę i wydłużają optycznie sylwetkę.',
    styles: ['Pompadour', 'Quiff'],
    image: `${API_URL}/images/male/pompadour.jpg`,
  },
  {
    face: 'Balanced proportions · High symmetry',
    face_pl: 'Zbalansowane proporcje · Wysoka symetria',
    analysis: 'Balanced proportions give you more freedom, so clean geometric cuts and classic styles work especially well.',
    analysis_pl: 'Zbalansowane proporcje dają większą swobodę, dlatego dobrze sprawdzają się klasyczne i geometryczne cięcia.',
    styles: ['Classic Undercut', 'Crew Cut'],
    image: `${API_URL}/images/male/classic_undercut.jpg`,
  },
]

const DEMO_CASES_WOMAN = [
  {
    face: 'Long face · Narrow jaw',
    face_pl: 'Długa twarz · Wąska szczęka',
    analysis: 'A bob, layers or curtain fringe adds width and breaks up the vertical proportions.',
    analysis_pl: 'Bob, warstwy lub grzywka dodają szerokości i przełamują pionowe proporcje twarzy.',
    styles: ['French Bob', 'Curtain Fringe Medium'],
    image: `${API_URL}/images/female/french_bob.jpg`,
  },
  {
    face: 'Wide jaw · Strong lower face',
    face_pl: 'Szeroka szczęka · Mocniejsza dolna część twarzy',
    analysis: 'Layers and length below the chin soften the jaw and create a longer, more flowing silhouette.',
    analysis_pl: 'Warstwy i długość poniżej brody łagodzą szeroką szczękę i tworzą bardziej wydłużoną, płynną sylwetkę.',
    styles: ['Layered Medium', 'Long Bob'],
    image: `${API_URL}/images/female/layered_medium.jpg`,
  },
  {
    face: 'Balanced thirds · High symmetry',
    face_pl: 'Zbalansowane proporcje · Wysoka symetria',
    analysis: 'Balanced proportions give you flexibility, making soft textures, waves and structured styles easy to wear.',
    analysis_pl: 'Zbalansowane proporcje dają większą swobodę, dlatego dobrze sprawdzają się fale, miękka tekstura i bardziej uporządkowane fryzury.',
    styles: ['Beach Waves', 'Classic Updo',],
    image: `${API_URL}/images/female/beach_waves.jpg`,
  },
]

const STEPS = [
  {
    num: '01',
    icon: '📸',
    title_en: 'Upload a photo',
    title_pl: 'Wrzuć zdjęcie',
    desc_en:  'A clear front-facing photo in good lighting - that\'s all needed.',
    desc_pl:  'Wyraźne zdjęcie przodem w dobrym oświetleniu - tylko tyle potrzeba.',
  },
  {
    num: '02',
    icon: '🔬',
    title_en: 'AI analyses your face',
    title_pl: 'AI analizuje Twoją twarz',
    desc_en:  'We measure 15 facial proportions using 478 landmark points.',
    desc_pl:  'Mierzymy 15 proporcji twarzy przy użyciu 478 punktów.',
  },
  {
    num: '03',
    icon: '✂️',
    title_en: 'Get your recommendations',
    title_pl: 'Otrzymaj spersonalizowane wskazówki',
    desc_en:  'Personalised hairstyles ranked by how well they suit your face geometry.',
    desc_pl:  'Fryzury dobrane specjalnie dla Ciebie uszeregowane według dopasowania.',
  },
]

const FEATURES = [
  { icon: '📐', title_en: 'Geometric analysis', title_pl: 'Analiza geometryczna', desc_en: '15 facial ratios measured against population norms', desc_pl: '15 proporcji twarzy zmierzonych względem norm populacyjnych' },
  { icon: '💇', title_en: 'Hair detection',  title_pl: 'Detekcja włosów', desc_en: 'Hair type and hairline shape detected automatically', desc_pl: 'Automatyczna detekcja typu i linii włosów' },
  { icon: '⚖️', title_en: 'Balance-first scoring', title_pl: 'Punktacja balansująca', desc_en: 'Styles scored on how well they balance and enhance your features', desc_pl: 'Fryzury oceniane pod kątem balansowania i uwydatniania Twoich rysów' },
  { icon: '🤖', title_en: 'AI face analysis', title_pl: 'Analiza AI', desc_en: 'Explanation of your facial strengths', desc_pl: 'Wyjaśnienie mocnych stron Twojej twarzy' },
]

export function Landing() {
  const { i18n } = useTranslation()
  const navigate  = useNavigate()
  const pl = i18n.language === 'pl'
  const [dark, setDark] = useState(false)

  function toggleLang() {
    const next = pl ? 'en' : 'pl'
    i18n.changeLanguage(next)
    localStorage.setItem('lang', next)
  }

  const t = (en, plStr) => pl ? plStr : en
  const [demoGender, setDemoGender] = useState('Man')
  const demoCases = demoGender === 'Man' ? DEMO_CASES_MAN : DEMO_CASES_WOMAN

  return (
    <div style={{minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)', 
    fontFamily: 'var(--font-body)'}} data-theme={dark ? 'dark' : 'light'}>

      {/* ── nav ── */}
      <nav style={{
        display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', padding: '18px 40px',
        borderBottom: '1px solid var(--border)', position: 'sticky',
        top: 0, background: 'var(--bg)', zIndex: 100,
      }}>
        <div style={{fontFamily: 'var(--font-display)', fontSize: 18,
          fontWeight: 500, letterSpacing: '.01em'}}>
          FaceFit <span style={{ color: 'var(--accent)' }}>AI</span>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button onClick={toggleLang} style={btnOutline}>
            {pl ? '🇬🇧 EN' : '🇵🇱 PL'}
          </button>
          <button onClick={() => setDark(d => !d)} style={btnOutline}>
            {dark ? '☀️' : '🌙'}
          </button>
          <button
            onClick={() => navigate('/analyse')}
            style={btnAccent}
          >
            {t('Try it', 'Wypróbuj')}
          </button>
        </div>
      </nav>

      {/* ── hero ── */}
      <section style={{
        maxWidth:  780,
        margin: '0 auto',
        padding: '80px 24px 64px',
        textAlign: 'center',
      }}>
        <div style={{
          display: 'inline-block', fontSize: 11, fontFamily: 'var(--font-mono)',
          letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--accent)',
          background: 'var(--accent-soft)', padding: '4px 12px', borderRadius: 20,
          marginBottom: 24, border: '1px solid var(--accent)',
        }}>
          {t('AI-Powered · Free to try', 'Oparte na AI · Przetestuj za darmo')}
        </div>

        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontSize: 'clamp(32px, 5vw, 52px)',
          fontWeight: 500,
          lineHeight: 1.15,
          letterSpacing: '.01em',
          marginBottom: 20,
          color: 'var(--text)',
        }}>
          {t(
            'Find hairstyles that actually suit your face',
            'Znajdź fryzurę która realnie pasuje do Twojej twarzy'
          )}
        </h1>

        <p style={{
          fontSize: 'clamp(15px, 2vw, 18px)', color: 'var(--text-muted)',
          fontWeight: 300, lineHeight: 1.65, maxWidth: 560, margin: '0 auto 36px',
        }}>
          {t(
            'Upload a photo and get science-backed hairstyle recommendations based on the geometry of your face.',
            'Wrzuć zdjęcie i otrzymaj propozycje fryzur oparte na geometrii Twojej twarzy.'
          )}
        </p>

        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
          <button onClick={() => navigate('/analyse')} style={{ ...btnAccent, fontSize: 15, padding: '13px 32px' }}>
            {t('Analyse my face →', 'Analizuj moją twarz →')}
          </button>
          <a href="#how-it-works" style={{ ...btnOutline, fontSize: 15, padding: '13px 32px', textDecoration: 'none' }}>
            {t('See how it works', 'Zobacz jak to działa')}
          </a>
        </div>

        {/* trust line */}
        <p style={{ marginTop: 28, fontSize: 12, color: 'var(--text-hint)', fontFamily: 'var(--font-mono)' }}>
          {t('478 landmarks · 15 facial ratios · free · no account needed',
             '478 punktów · 15 proporcji twarzy · bezpłatne · bez rejestracji')}
        </p>
      </section>

      {/* ── how it works ── */}
      <section id="how-it-works" style={{
        background: 'var(--surface)',
        borderTop: '1px solid var(--border)',
        borderBottom: '1px solid var(--border)',
        padding: '64px 24px',
      }}>
        <div style={{ maxWidth: 880, margin: '0 auto' }}>
          <SectionLabel text={t('How it works', 'Jak to działa')} />
          <h2 style={sectionH2}>
            {t('Three steps to your perfect hairstyle', 'Trzy kroki do otrzymania Twojej idealnej fryzury')}
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 24, marginTop: 40 }}>
            {STEPS.map(s => (
              <div key={s.num} style={{
                background: 'var(--bg)', borderRadius: 'var(--radius-lg)',
                border: '1px solid var(--border)', padding: '24px 22px',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                  <span style={{
                    fontFamily: 'var(--font-mono)', fontSize: 11,
                    color: 'var(--accent)', letterSpacing: '.06em',
                  }}>{s.num}</span>
                  <span style={{ fontSize: 22 }}>{s.icon}</span>
                </div>
                <p style={{ fontSize: 15, fontWeight: 500, marginBottom: 8, color: 'var(--text)' }}>
                  {pl ? s.title_pl : s.title_en}
                </p>
                <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.6 }}>
                  {pl ? s.desc_pl : s.desc_en}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── features ── */}
      <section style={{ padding: '64px 24px' }}>
        <div style={{ maxWidth: 880, margin: '0 auto' }}>
          <SectionLabel text={t('What you get', 'Co otrzymujesz')} />
          <h2 style={sectionH2}>
            {t('More than a style quiz', 'Więcej niż quiz o stylach')}
          </h2>
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: 16, marginTop: 40,
          }}>
            {FEATURES.map(f => (
              <div key={f.icon} style={{
                background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
                border: '1px solid var(--border)', padding: '20px',
              }}>
                <span style={{ fontSize: 24, display: 'block', marginBottom: 12 }}>{f.icon}</span>
                <p style={{ fontSize: 13, fontWeight: 500, marginBottom: 6, color: 'var(--text)' }}>
                  {pl ? f.title_pl : f.title_en}
                </p>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.55 }}>
                  {pl ? f.desc_pl : f.desc_en}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── demo ── */}
      <section style={{
        background: 'var(--surface)', borderTop: '1px solid var(--border)',
        borderBottom: '1px solid var(--border)', padding: '64px 24px'
      }}>
        <div style={{ maxWidth: 880, margin: '0 auto' }}>
          <SectionLabel text={t('Example results', 'Przykładowe wyniki')} />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: 12, marginBottom: 8 }}>
            <h2 style={sectionH2}>
              {t('See what the system recommends', 'Zobacz co doradzi aplikacja')}
            </h2>
            {/* gender toggle */}
            <div style={{ display: 'flex', gap: 6 }}>
              {['Man', 'Woman'].map(g => (
                <button
                  key={g}
                  onClick={() => setDemoGender(g)}
                  style={{
                    ...btnOutline,
                    borderColor:   demoGender === g ? 'var(--accent)' : 'var(--border)',
                    color:         demoGender === g ? 'var(--accent)' : 'var(--text-muted)',
                    background:    demoGender === g ? 'var(--accent-soft)' : 'none',
                  }}
                >
                  {g === 'Man'
                    ? t('👨 Men', '👨 Mężczyźni')
                    : t('👩 Women', '👩 Kobiety')}
                </button>
              ))}
            </div>
        </div>

        <div 
        className="demo-grid"
        style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20, marginTop: 24}}>
          {demoCases.map((c, i) => (
            <div key={`${demoGender}-${i}`} style={{
              background: 'var(--bg)', borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border)', overflow: 'hidden',
              animation: 'fadeIn .3s ease',
            }}>
              {/* image */}
              <img className="card-mobile"
                src={c.image}
                alt={c.face}
                style={{
                  width: '100%', height: 400,
                  objectFit: 'cover', objectPosition: 'top',
                  display: 'block', borderBottom: '1px solid var(--border)',
                }}
                onError={e => {
                  e.target.style.display = 'none'
                  e.target.nextSibling.style.display = 'flex'
                }}
              />
              {/* fallback */}
              <div style={{
                height: 200, background: 'var(--surface-2)',
                display: 'none', alignItems: 'center', justifyContent: 'center',
                fontSize: 48, borderBottom: '1px solid var(--border)',
              }}>✂️</div>

              <div style={{ padding: '16px' }}>
                <div style={{
                  fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--accent)',
                  letterSpacing: '.06em', textTransform: 'uppercase', marginBottom: 8,
                }}>{pl ? c.face_pl : c.face}</div>

              <p style={{
                fontSize: 12, color: 'var(--text-muted)', fontWeight: 300,
                lineHeight: 1.6, marginBottom: 14,
                paddingLeft: 8, borderLeft: '2px solid var(--accent)',
              }}>{pl ? c.analysis_pl : c.analysis}</p>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, justifyContent: 'center'}}>
                {c.styles.map(s => (
                  <span key={s} style={{
                    fontSize: 10, padding: '3px 9px', borderRadius: 20,
                    background: 'var(--surface)', border: '1px solid var(--border)',
                    color: 'var(--text-muted)', letterSpacing: '.03em',
                  }}>{s}</span>
                ))}
              </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>

      {/* ── for professionals ── */}
      <section style={{ padding: '64px 24px' }}>
        <div style={{
          maxWidth: 640, margin: '0 auto', textAlign: 'center', background: 'var(--surface)',
          borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)', padding: '40px 32px'
        }}>
          <span style={{ fontSize: 32, display: 'block', marginBottom: 16 }}>✂️</span>
          <h2 style={{ ...sectionH2, marginBottom: 12 }}>
            {t('For hairstylists & salons', 'Dla fryzjerów i salonów')}
          </h2>
          <p style={{
            fontSize: 14, color: 'var(--text-muted)', fontWeight: 300,
            lineHeight: 1.7, marginBottom: 24
          }}>
            {t(
              'Use FaceFit AI as a consultation tool — show clients data-driven style suggestions before the cut. Currently in beta, contact us for early access.',
              'Używaj FaceFit AI jako narzędzia konsultacyjnego, zaproponuj klientom analizę fryzur opartą na danych przed strzyżeniem. Aktualnie w fazie beta, skontaktuj się po wcześniejszy dostęp.'
            )}
          </p>
          
            <a href="mailto:jurewiczjuras@gmail.com"
            style={{ ...btnAccent, textDecoration: 'none', display: 'inline-block' }}
          >
            {t('Contact us', 'Skontaktuj się')}
          </a>
        </div>
      </section>

      {/* ── cta footer ── */}
      <section style={{
        background: 'var(--surface)', borderTop: '1px solid var(--border)',
        padding: '64px 24px', textAlign: 'center'}}>
        <h2 style={{ ...sectionH2, marginBottom: 12 }}>
          {t('Ready to find your style?', 'Gotowy żeby odkryć idealną fryzurę dla siebie?')}
        </h2>
        <p style={{ fontSize: 14, color: 'var(--text-muted)', fontWeight: 300, marginBottom: 28}}>
          {t('Free · No account · Results in seconds', 'Bezpłatne · Bez rejestracji · Wyniki w kilka sekund')}
        </p>
        <button
          onClick={() => navigate('/analyse')}
          style={{ ...btnAccent, fontSize: 15, padding: '13px 36px' }}
        >
          {t('Analyse my face →', 'Analizuj moją twarz →')}
        </button>
      </section>

      {/* ── footer ── */}
      <footer style={{
        borderTop: '1px solid var(--border)', padding: '20px 40px', display: 'flex',
        justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8}}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: 14, color: 'var(--text-muted)' }}>
          FaceFit <span style={{ color: 'var(--accent)' }}>AI</span>
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-hint)', fontFamily: 'var(--font-mono)' }}>
          {t('Built with MediaPipe · Python · React', 'Zbudowane z MediaPipe · Python · React')}
        </span>
      </footer>
    </div>
  )
}

/* ── shared styles ── */
const btnAccent = {
  background: 'var(--accent)',
  color: '#fff',
  border: 'none',
  borderRadius: 'var(--radius-md)',
  padding: '10px 20px',
  fontSize: 13,
  fontFamily: 'var(--font-body)',
  fontWeight: 500,
  cursor: 'pointer',
  letterSpacing: '.02em',
  transition: 'background .15s',
}

const btnOutline = {
  background: 'none',
  color: 'var(--text-muted)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-sm)',
  padding: '7px 14px',
  fontSize: 12,
  fontFamily: 'var(--font-body)',
  cursor: 'pointer',
  letterSpacing: '.02em',
  transition: 'border-color .15s',
}

const sectionH2 = {
  fontFamily: 'var(--font-display)',
  fontSize: 'clamp(22px, 3vw, 30px)',
  fontWeight: 500,
  letterSpacing: '.01em',
  color: 'var(--text)',
  marginBottom: 8,
}

function SectionLabel({ text }) {
  return (
    <div style={{
      fontSize: 10,
      fontFamily: 'var(--font-mono)',
      letterSpacing: '.1em',
      textTransform: 'uppercase',
      color: 'var(--accent)',
      marginBottom: 10,
    }}>{text}</div>
  )
}