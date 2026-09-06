import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { useDarkMode } from '../hooks/useDarkMode'
import { useAuth } from '../hooks/useAuth'
import { History, Sparkles, Camera, LogOut, User, Languages, ChevronDown, Scissors } from 'lucide-react'
import { btnAccent, btnOutline, darkToggleBtn, darkToggleTrack, darkToggleKnob } from '../styles/shared'
import { AuthModal } from './AuthModal'


export function NavBar({
  variant = 'landing',
  onShowPanel,
  onShowTutorial,
  onUpgrade,
  isPremium = false,
}) {
  const { i18n } = useTranslation()
  const navigate = useNavigate()
  const pl = i18n.language === 'pl'
  const [dark, setDark] = useDarkMode()
  const { user, signOut } = useAuth()

  const [showAuth, setShowAuth] = useState(false)
  const [showUserMenu, setShowUserMenu] = useState(false)

  function toggleLang() {
    const next = pl ? 'en' : 'pl'
    i18n.changeLanguage(next)
    localStorage.setItem('lang', next)
  }

  const t = (en, plStr) => pl ? plStr : en

  /* menu items - shared between logged-in users on both pages */
  const menuItems = [
    variant === 'app' && {
      Icon: History,
      label: t('Analysis history', 'Historia analiz'),
      action: () => { onShowPanel?.(); setShowUserMenu(false) },
    },
    !isPremium && {
      Icon: Sparkles,
      label: t('Get Premium', 'Kup Premium'),
      action: () => { onUpgrade?.(); setShowUserMenu(false) },
      accent: true,
    },
    variant === 'app' && {
      Icon: Camera,
      label: t('Photo tips', 'Wskazówki zdjęciowe'),
      action: () => { onShowTutorial?.(); setShowUserMenu(false) },
    },
    {
      Icon: Languages,
      label: pl ? 'English' : 'Polski',
      action: () => { toggleLang(); setShowUserMenu(false) },
    },
    {
      Icon: LogOut,
      label: t('Sign out', 'Wyloguj'),
      action: () => { signOut(); setShowUserMenu(false) },
    },
  ].filter(Boolean)

  /* guest menu — for non-logged-in users */
  const guestMenuItems = [
    {
      Icon: Languages,
      label: pl ? 'English' : 'Polski',
      action: () => { toggleLang(); setShowUserMenu(false) },
    },
    variant === 'app' && {
      Icon: Camera,
      label: t('Photo tips', 'Wskazówki zdjęciowe'),
      action: () => { onShowTutorial?.(); setShowUserMenu(false) },
    },
    {
      Icon: User,
      label: t('Sign in', 'Zaloguj się'),
      action: () => { setShowAuth(true); setShowUserMenu(false) },
      accent: true,
    },
  ].filter(Boolean)

  const activeMenuItems = user ? menuItems : guestMenuItems

  return (
    <>
      <nav className="site-nav" style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '16px 32px', borderBottom: '1px solid var(--border)',
        position: 'sticky', top: 0, background: 'var(--bg)', zIndex: 100,
      }}>
        {/* logo */}
        <div
          onClick={() => navigate(variant === 'app' ? '/' : '/')}
          style={{
            fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 500,
            letterSpacing: '.01em', cursor: 'pointer',display: 'flex',
            alignItems: 'center', gap: 8, color: 'var(--text)',
          }}
        >
          <img src="/android-chrome-192x192.png" alt=""
            style={{ width: 22, height: 22, borderRadius: 4 }} />
          Stylizzer
        </div>

        {/* right side */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>

          {/* dark toggle */}
          <button onClick={() => setDark(d => !d)} style={darkToggleBtn(dark)}>
            <span style={{ fontSize: 12 }}>{dark ? '☀️' : '🌙'}</span>
            <span style={darkToggleTrack(dark)}>
              <span style={darkToggleKnob(dark)} />
            </span>
          </button>

          {/* lang button - visible on desktop, hidden on mobile (in menu) */}
          <button
            onClick={toggleLang}
            className="nav-lang-btn"
            style={btnOutline}
          >
            {pl ? 'EN' : 'PL'}
          </button>

          {/* separator */}
          <div style={{ width: 1, height: 16, background: 'var(--border)' }} />

          {/* user menu or sign in */}
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setShowUserMenu(m => !m)}
              style={{
                ...btnOutline, display: 'flex', alignItems: 'center',
                gap: 6, maxWidth: 160,
              }}
            >
              {user ? (
                <>
                  <div style={{
                    width: 18, height: 18, borderRadius: '50%',
                    background: isPremium ? 'var(--accent)' : 'var(--border)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 9, color: '#fff', fontWeight: 600, flexShrink: 0,
                  }}>
                    {user.email?.[0]?.toUpperCase()}
                  </div>
                  <span style={{
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 90,
                  }}>
                    {user.email?.split('@')[0]}
                  </span>
                </>
              ) : (
                <>
                  <User size={13} strokeWidth={1.5} />
                  <span>{t('Account', 'Konto')}</span>
                </>
              )}
              <ChevronDown
                size={11}
                style={{
                  transform:  showUserMenu ? 'rotate(180deg)' : 'none',
                  transition: 'transform .2s',
                  flexShrink: 0,
                }}
              />
            </button>

            {showUserMenu && (
              <>
                <div
                  onClick={() => setShowUserMenu(false)}
                  style={{ position: 'fixed', inset: 0, zIndex: 98 }}
                />
                <div style={{
                  position: 'absolute', top: 'calc(100% + 6px)', right: 0, zIndex: 99,
                  background: 'var(--surface)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)', boxShadow: '0 8px 24px rgba(0,0,0,.15)',
                  minWidth: 190, overflow: 'hidden', animation: 'fadeIn .15s ease',
                }}>
                  {/* email/status header */}
                  {user && (
                    <div style={{
                      padding: '10px 14px', borderBottom: '1px solid var(--border)',
                      background: 'var(--surface-2)',
                    }}>
                      <p style={{
                        fontSize: 11, color: 'var(--text-muted)', fontWeight: 300,
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      }}>
                        {user.email}
                      </p>
                      <p style={{
                        fontSize: 10, fontFamily: 'var(--font-mono)', marginTop: 2,
                        color: isPremium ? 'var(--accent)' : 'var(--text-hint)',
                      }}>
                        {isPremium ? '✦ Premium' : '○ Free'}
                      </p>
                    </div>
                  )}

                  {/* menu items */}
                  {activeMenuItems.map(item => {
                    const ItemIcon = item.Icon
                    return (
                      <button
                        key={item.label}
                        onClick={item.action}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 10,
                          width: '100%', padding: '9px 14px', background: 'none', border: 'none',
                          cursor: 'pointer', fontSize: 12, textAlign: 'left',
                          color: item.accent ? 'var(--accent)' : 'var(--text)',
                          fontFamily: 'var(--font-body)', transition: 'background .1s',
                        }}
                        onMouseEnter={e => e.currentTarget.style.background = 'var(--surface-2)'}
                        onMouseLeave={e => e.currentTarget.style.background = 'none'}
                      >
                        <ItemIcon
                          size={14}
                          strokeWidth={1.5}
                          color={item.accent ? 'var(--accent)' : 'var(--text-muted)'}
                        />
                        {item.label}
                      </button>
                    )
                  })}
                </div>
              </>
            )}
          </div>

          {/* CTA - landing */}
          {variant === 'landing' && (
            <button
              onClick={() => navigate('/analyse')}
              style={{
                ...btnAccent, display: 'inline-flex', alignItems: 'center', gap: 6,
              }}
            >
              {t('Try it', 'Przetestuj')}
              <Scissors size={13} strokeWidth={1.5} />
            </button>
          )}
        </div>
      </nav>

      {showAuth && (
        <AuthModal
          onClose={() => setShowAuth(false)}
          onSuccess={() => setShowAuth(false)}
        />
      )}
    </>
  )
}