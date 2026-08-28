import { useTranslation } from 'react-i18next'

export function PremiumGate({ children, isPremium, onUnlock }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'

  if (isPremium) return children

  return (
    <div style={{ position: 'relative' }}>
      {/* blurred */}
      <div style={{
        filter: 'blur(4px)', pointerEvents: 'none', userSelect: 'none', opacity: 0.6,
      }}>
        {children}
      </div>

      {/* overlay */}
      <div
        onClick={onUnlock}
        style={{
          position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
          borderRadius: 'var(--radius-lg)', background: 'rgba(15,15,14,.45)',
          backdropFilter: 'blur(2px)', gap: 10,
        }}
      >
        <span style={{ fontSize: 28 }}>✨</span>
        <p style={{
          fontFamily: 'var(--font-display)', fontSize: 15, fontWeight: 500,
          color: '#fff', textAlign: 'center', padding: '0 24px',
        }}>
          {pl ? 'Funkcja premium' : 'Premium feature'}
        </p>
        <p style={{
          fontSize: 12, color: 'rgba(255,255,255,.7)', textAlign: 'center', 
          padding: '0 24px', fontWeight: 300,
        }}>
          {pl ? 'Kliknij aby odblokować' : 'Click to unlock'}
        </p>
      </div>
    </div>
  )
}