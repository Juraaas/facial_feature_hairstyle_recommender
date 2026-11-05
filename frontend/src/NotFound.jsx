import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useDarkMode } from './hooks/useDarkMode'

export function NotFound() {
  const { i18n } = useTranslation()
  const pl = i18n.language?.startsWith('pl')
  const [dark] = useDarkMode()

  return (
    <div data-theme={dark ? 'dark' : 'light'} style={{
      minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)',
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', gap: 12, padding: 24, textAlign: 'center',
    }}>
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-hint)' }}>404</p>
      <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 28, fontWeight: 500 }}>
        {pl ? 'Nie znaleziono strony' : 'Page not found'}
      </h1>
      <p style={{ fontSize: 14, color: 'var(--text-muted)', fontWeight: 300 }}>
        {pl ? 'Adres, który otworzyłeś, nie istnieje.' : "The address you opened doesn't exist."}
      </p>
      <Link to="/" style={{ color: 'var(--accent)', fontSize: 14, textDecoration: 'none' }}>
        {pl ? '← Wróć na stronę główną' : '← Back to home'}
      </Link>
    </div>
  )
}