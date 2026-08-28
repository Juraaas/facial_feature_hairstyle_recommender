import { useTranslation } from 'react-i18next'

export function PremiumPopup({ onClose, onUpgrade, onLogin, user }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'

  return (
    <>
      <div onClick={onClose} style={{
        position: 'fixed', inset: 0, 
        background: 'rgba(0,0,0,.5)', zIndex: 200,
      }} />
      <div style={{
        position: 'fixed', top: '50%', left: '50%',transform: 'translate(-50%, -50%)',
        zIndex: 201, width: '100%', maxWidth: 380,background: 'var(--surface)',
        borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)',
        padding: '32px 28px', textAlign: 'center', animation: 'fadeIn .2s ease',
      }}>
        <button onClick={onClose} style={{
          position: 'absolute', top: 14, right: 14,
          background: 'none', border: 'none',
          cursor: 'pointer', fontSize: 18, color: 'var(--text-hint)',
        }}>✕</button>

        <div style={{
          width: 52, height: 52, borderRadius: '50%',
          background: 'var(--accent-soft)', border: '1.5px solid var(--accent)',
          display: 'flex', alignItems: 'center',
          justifyContent: 'center', fontSize: 22, margin: '0 auto 16px',
        }}>✨</div>

        <h3 style={{
          fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 500,
          color: 'var(--text)', marginBottom: 8,
        }}>
          {pl ? 'Odblokuj FaceFit Premium' : 'Unlock FaceFit Premium'}
        </h3>

        <p style={{fontSize: 13, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.6, marginBottom: 20}}>
          {pl
            ? 'Analiza AI, szczegółowe wyjaśnienia, podgląd fryzur i historia analiz.'
            : 'AI face analysis, detailed explanations, style preview and analysis history.'}
        </p>

        {/* feature list */}
        <div style={{
          textAlign: 'left', marginBottom: 24, display: 'flex', flexDirection: 'column', gap: 8
        }}>
          {[
            pl ? '✓ Analiza AI' : '✓ AI-powered face analysis',
            pl ? '✓ Szczegółowe wyjaśnienia rekomendacji' : '✓ Detailed style explanations',
            pl ? '✓ Podgląd fryzur' : '✓ Hairstyle preview',
            pl ? '✓ Historia analiz' : '✓ Full analysis history',
          ].map(f => (
            <span key={f} style={{fontSize: 12, color: 'var(--text)',fontWeight: 300}}>{f}</span>
          ))}
        </div>

        <button onClick={onUpgrade} className="analyse-btn" style={{ marginBottom: 10 }}>
          {pl ? 'Ulepsz do premium →' : 'Get premium access →'}
        </button>

        {!user && (
          <button onClick={onLogin} style={{
            background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: 12,
            cursor: 'pointer', fontFamily: 'var(--font-body)',
          }}>
            {pl ? 'Mam już konto - zaloguj mnie' : 'I have an account - sign in'}
          </button>
        )}
      </div>
    </>
  )
}