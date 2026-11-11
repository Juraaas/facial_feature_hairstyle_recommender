import { useTranslation } from 'react-i18next'
import { Coins } from 'lucide-react'
import { creditWord } from '../lib/credits'

export function CreditsBar({ credits, costs, onBuy }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
      padding: '8px 14px', marginBottom: 16, background: 'var(--surface)',
      border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
    }}>
      <Coins size={15} color="var(--accent)" strokeWidth={1.5} />
      <span style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>
        {credits} {creditWord(credits, pl)}
      </span>
      <span style={{ fontSize: 11, color: 'var(--text-hint)', flex: 1, fontWeight: 300 }}>
        {pl ? `analiza ${costs.analysis} · podgląd ${costs.preview}`
            : `analysis ${costs.analysis} · preview ${costs.preview}`}
      </span>
      <button onClick={onBuy} style={{
        fontSize: 12, padding: '5px 14px', borderRadius: 20, cursor: 'pointer',
        border: '1px solid var(--accent)', background: 'var(--accent-soft)',
        color: 'var(--accent)', fontFamily: 'var(--font-body)',
      }}>{pl ? 'Kup kredyty' : 'Buy credits'}</button>
    </div>
  )
}