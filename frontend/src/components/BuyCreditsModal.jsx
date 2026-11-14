import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { X } from 'lucide-react'
import { creditWord, plural } from '../lib/credits'

const CSS = `
.bc-overlay { position: fixed; inset: 0; z-index: 400; background: rgba(15,15,14,.55);
  display: flex; align-items: center; justify-content: center; padding: 16px; }
.bc-sheet { position: relative; width: 100%; max-width: 760px;
  max-height: calc(100vh - 32px); max-height: calc(100dvh - 32px);
  overflow-y: auto; -webkit-overflow-scrolling: touch; overscroll-behavior: contain;
  background: var(--bg); border: 1px solid var(--border); border-radius: 16px; padding: 26px 22px 22px; }
.bc-grab { display: none; }
.bc-close { position: absolute; top: 12px; right: 12px; background: none; border: none;
  cursor: pointer; color: var(--text-hint); display: flex; padding: 6px; }
.bc-title { font-family: var(--font-display); font-size: 20px; font-weight: 500; margin: 0 28px 6px 0; }
.bc-lead { font-size: 13px; color: var(--text-muted); font-weight: 300; line-height: 1.5; margin: 0; }
.bc-consent { display: flex; gap: 10px; align-items: flex-start; cursor: pointer; margin: 16px 0 4px;
  padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
.bc-consent input { margin-top: 3px; flex-shrink: 0; width: 16px; height: 16px; }
.bc-consent span { font-size: 11px; color: var(--text-muted); line-height: 1.6; font-weight: 300; }
.bc-consent a { color: var(--accent); }
.bc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px; margin: 18px 0 8px; }
.bc-card { position: relative; display: flex; flex-direction: column; gap: 4px; text-align: center;
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px 16px; }
.bc-featured { border: 1.5px solid var(--accent); }
.bc-badge { position: absolute; top: -10px; left: 50%; transform: translateX(-50%);
  background: var(--accent); color: #fff; font-size: 9px; font-weight: 600; padding: 3px 10px;
  border-radius: 20px; letter-spacing: .06em; text-transform: uppercase; white-space: nowrap; }
.bc-name { font-size: 12px; color: var(--text-muted); margin: 0; }
.bc-credits { font-family: var(--font-mono); font-size: 30px; font-weight: 700; color: var(--accent);
  line-height: 1; margin: 2px 0; }
.bc-sub { font-size: 11px; color: var(--text-hint); margin: 0; }
.bc-buy { display: flex; flex-direction: column; gap: 2px; align-items: center; margin-top: 6px; }
.bc-price { font-size: 20px; font-weight: 600; margin: 0; }
.bc-per { font-size: 11px; color: var(--text-hint); margin: 0 0 8px; }
.bc-card .bc-btn { margin-top: 0; }
.bc-foot { font-size: 10px; color: var(--text-hint); margin: 10px 0 0; text-align: center; }

@media (max-width: 640px) {
  .bc-overlay { align-items: flex-end; padding: 0; }
  .bc-sheet { max-width: none; max-height: 92vh; max-height: 92dvh; border-radius: 18px 18px 0 0;
    border-bottom: none; padding: 10px 16px calc(16px + env(safe-area-inset-bottom)); }
  .bc-grab { display: block; width: 36px; height: 4px; border-radius: 2px; background: var(--border);
    margin: 0 auto 14px; }
  .bc-close { top: 18px; }
  .bc-title { font-size: 18px; }
  .bc-grid { grid-template-columns: 1fr; gap: 10px; margin-top: 14px; }
  .bc-card { flex-direction: row; align-items: center; justify-content: space-between;
    text-align: left; gap: 12px; padding: 12px 14px; }
  .bc-badge { top: -9px; left: 14px; transform: none; }
  .bc-credits { font-size: 24px; }
  .bc-buy { align-items: flex-end; margin-top: 0; flex-shrink: 0; }
  .bc-price { font-size: 17px; }
  .bc-per { margin-bottom: 6px; }
  .bc-card .bc-btn { width: auto; min-width: 96px; padding: 8px 18px; font-size: 13px; }
}
`

export function BuyCreditsModal({ packs, costs, credits, buying, onBuy, onClose }) {
  const { i18n } = useTranslation()
  const pl = i18n.language?.startsWith('pl')
  const [agree, setAgree] = useState(false)

  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = prev }
  }, [])

  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const money = (cents, cur) =>
    new Intl.NumberFormat(pl ? 'pl-PL' : 'en-US',
      { style: 'currency', currency: (cur || 'pln').toUpperCase() }).format(cents / 100)

  return (
    <div className="bc-overlay" onClick={onClose}>
      <div className="bc-sheet" role="dialog" aria-modal="true"
        aria-label={pl ? 'Kup kredyty' : 'Buy credits'} onClick={e => e.stopPropagation()}>
        <div className="bc-grab" aria-hidden="true" />
        <button className="bc-close" onClick={onClose} aria-label="close">
          <X size={20} strokeWidth={1.5} />
        </button>

        <h2 className="bc-title">{pl ? 'Kup kredyty' : 'Buy credits'}</h2>
        <p className="bc-lead">
          {pl
            ? `Masz ${credits} ${creditWord(credits, true)}. Pełna analiza kosztuje ${costs.analysis}, podgląd fryzury ${costs.preview}.`
            : `You have ${credits} ${creditWord(credits, false)}. A full analysis costs ${costs.analysis}, a hairstyle preview ${costs.preview}.`}
        </p>

        {/* TODO zgody*/}
        <label className="bc-consent">
          <input type="checkbox" checked={agree} onChange={e => setAgree(e.target.checked)} />
          <span>
            {pl ? (
              <>Akceptuję <a href="/terms" target="_blank" rel="noopener noreferrer">Regulamin</a> i <a href="/privacy" target="_blank" rel="noopener noreferrer">Politykę Prywatności</a>. Zgadzam się na natychmiastowe dostarczenie kredytów i przyjmuję do wiadomości, że po ich wykorzystaniu nie przysługuje mi prawo odstąpienia od umowy.</>
            ) : (
              <>I accept the <a href="/terms" target="_blank" rel="noopener noreferrer">Terms</a> and <a href="/privacy" target="_blank" rel="noopener noreferrer">Privacy Policy</a>. I agree to immediate delivery of the credits and acknowledge that once they are used I lose my right of withdrawal.</>
            )}
          </span>
        </label>

        {packs.length === 0 ? (
          <p style={{ fontSize: 13, color: 'var(--text-hint)', margin: '24px 0' }}>…</p>
        ) : (
          <div className="bc-grid">
            {packs.map((p, i) => {
              const featured = packs.length === 3 && i === 1
              const previews = Math.floor(p.credits / costs.preview)
              return (
                <div key={p.id} className={`bc-card${featured ? ' bc-featured' : ''}`}>
                  {featured && <span className="bc-badge">{pl ? 'Najpopularniejszy' : 'Most popular'}</span>}

                  <div className="bc-main">
                    <p className="bc-name">{pl ? p.name_pl : p.name_en}</p>
                    <p className="bc-credits">{p.credits}</p>
                    <p className="bc-sub">
                      {pl
                        ? `${creditWord(p.credits, true)} · ok. ${previews} ${plural(previews, 'podgląd', 'podglądy', 'podglądów')}`
                        : `credits · about ${previews} ${previews === 1 ? 'preview' : 'previews'}`}
                    </p>
                  </div>

                  <div className="bc-buy">
                    <p className="bc-price">{money(p.price_cents, p.currency)}</p>
                    <p className="bc-per">
                      {pl
                        ? `ok. ${money(p.price_cents / p.credits * costs.preview, p.currency)} za podgląd`
                        : `about ${money(p.price_cents / p.credits * costs.preview, p.currency)} per preview`}
                    </p>
                    <button className="analyse-btn bc-btn" disabled={!agree || buying !== null}
                      onClick={() => onBuy(p.id)} style={{ opacity: agree ? 1 : 0.45 }}>
                      {buying === p.id ? '…' : (pl ? 'Kup' : 'Buy')}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        )}

        <p className="bc-foot">
          {pl ? 'Płatność obsługuje Stripe. Dane karty nie trafiają do nas.'
              : 'Payments are processed by Stripe. We never see your card details.'}
        </p>
      </div>
      <style>{CSS}</style>
    </div>
  )
}