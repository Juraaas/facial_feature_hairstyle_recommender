import { useState, useEffect } from 'react'
import { StyleCard } from './StyleCard'
import { useTranslation } from 'react-i18next'
import { ArrowRight, Sparkles, Lock } from 'lucide-react'

const TEASER_SLOTS = 2

function useIsDesktop() {
  const [v, setV] = useState(() => window.innerWidth >= 720)
  useEffect(() => {
    const on = () => setV(window.innerWidth >= 720)
    window.addEventListener('resize', on)
    return () => window.removeEventListener('resize', on)
  }, [])
  return v
}

function LockedStyleCard({ rank, pl, onUnlock }) {
  const bar = (w, h = 8) => <div style={{ width: w, height: h, borderRadius: 4, background: 'var(--border)' }} />
  return (
    <button type="button" onClick={onUnlock}
      aria-label={pl ? 'Odblokuj pełną analizę' : 'Unlock the full analysis'}
      style={{
        background: 'var(--surface)', borderRadius: 'var(--radius-lg)', overflow: 'hidden',
        border: '1px solid var(--border)', display: 'flex', flexDirection: 'column',
        padding: 0, cursor: 'pointer', textAlign: 'left', width: '100%',
      }}>
      <div className="style-card-img" style={{
        position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'linear-gradient(135deg, var(--surface-2), var(--border))',
      }}>
        <div style={{
          width: 44, height: 44, borderRadius: '50%', background: 'var(--surface)',
          border: '1.5px solid var(--accent)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Lock size={18} color="var(--accent)" strokeWidth={1.5} />
        </div>
        <span style={{
          position: 'absolute', top: 10, left: 10, fontSize: 10, fontFamily: 'var(--font-mono)',
          color: 'var(--text-hint)', background: 'var(--surface)', padding: '2px 8px',
          borderRadius: 20, border: '1px solid var(--border)',
        }}>#{rank}</span>
      </div>
      <div aria-hidden="true" style={{
        padding: '14px 14px 10px', display: 'flex', flexDirection: 'column', gap: 10,
        filter: 'blur(2px)', opacity: 0.6,
      }}>
        <div style={{ display: 'flex', justifyContent: 'center' }}>{bar('55%', 12)}</div>
        <div style={{ display: 'flex', gap: 6, justifyContent: 'center' }}>{bar(50, 14)}{bar(60, 14)}</div>
        {bar('100%')}{bar('92%')}{bar('70%')}
      </div>
      <p style={{
        textAlign: 'center', fontSize: 11, color: 'var(--accent)', fontWeight: 500, padding: '0 14px 14px',
      }}>
        {pl ? 'Dostępne w pełnej analizie' : 'Included in the full analysis'}
      </p>
    </button>
  )
}

export function StylesSection({ styles, lockedCount = 0, features, gender, unlocked, onUnlock }) {
  const { t, i18n } = useTranslation()
  const pl = i18n.language === 'pl'
  const isDesktop = useIsDesktop()
  const perPage = isDesktop ? 3 : 1
  const [page, setPage] = useState(0)

  const teaser = unlocked ? 0 : Math.min(lockedCount, TEASER_SLOTS)
  const items = [
    ...styles.map(style => ({ locked: false, style })),
    ...Array.from({ length: teaser }, () => ({ locked: true })),
  ]
  const totalPages = Math.max(1, Math.ceil(items.length / perPage))
  const start = page * perPage
  const visible = items.slice(start, start + perPage)

  useEffect(() => { setPage(0) }, [styles, i18n.language, perPage])

  function prev() { setPage(p => Math.max(0, p - 1)) }
  function next() { setPage(p => Math.min(totalPages - 1, p + 1)) }

  return (
    <section style={{ marginBottom: 32 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2 className="section-title">{t('section_hairstyles')}</h2>
        {totalPages > 1 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-hint)' }}>
              {page + 1} / {totalPages}
            </span>
            <button onClick={prev} disabled={page === 0} style={navBtn(page === 0)}>←</button>
            <button onClick={next} disabled={page === totalPages - 1} style={navBtn(page === totalPages - 1)}>→</button>
          </div>
        )}
      </div>

      <div className="styles-grid" style={{ animation: 'fadeIn .25s ease' }} key={page}>
        {visible.map((item, i) => item.locked
          ? <LockedStyleCard key={`locked-${page}-${i}`} rank={start + i + 1} pl={pl} onUnlock={onUnlock} />
          : <StyleCard key={`${item.style.name}-${page}-${i}`} style={item.style}
              rank={start + i} features={features} gender={gender} />)}
      </div>

      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'center', gap: 6, marginTop: 16 }}>
          {Array.from({ length: totalPages }).map((_, i) => (
            <button key={i} onClick={() => setPage(i)} aria-label={`${i + 1}`} style={{
              width: i === page ? 20 : 6, height: 6, borderRadius: 3, border: 'none', cursor: 'pointer',
              background: i === page ? 'var(--accent)' : 'var(--border)', transition: 'all .25s', padding: 0,
            }} />
          ))}
        </div>
      )}

      {!unlocked && lockedCount > 0 && (
        <div onClick={onUnlock} role="button" tabIndex={0}
          onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && onUnlock?.()}
          style={{
            marginTop: 16, padding: '12px 16px', background: 'var(--accent-soft)',
            border: '1px solid var(--accent)', borderRadius: 'var(--radius-md)',
            display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer',
          }}>
          <Sparkles size={16} color="var(--accent)" strokeWidth={1.5} />
          <div>
            <p style={{ fontSize: 12, fontWeight: 500, color: 'var(--accent)' }}>
              {pl ? `+${lockedCount} kolejnych rekomendacji` : `+${lockedCount} more recommendations`}
            </p>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 300 }}>
              {pl ? 'Odblokuj pełną analizę, aby zobaczyć wszystkie z wyjaśnieniami'
                  : 'Unlock the full analysis to see all of them with explanations'}
            </p>
          </div>
          <ArrowRight size={14} color="var(--accent)" style={{ marginLeft: 'auto' }} />
        </div>
      )}
    </section>
  )
}

function navBtn(disabled) {
  return {
    width: 32, height: 32, borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)',
    background: 'var(--surface)', cursor: disabled ? 'default' : 'pointer', fontSize: 14,
    color: disabled ? 'var(--text-hint)' : 'var(--text)', opacity: disabled ? 0.4 : 1,
    transition: 'all .15s', display: 'flex', alignItems: 'center', justifyContent: 'center',
  }
}