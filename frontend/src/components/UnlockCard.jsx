import { Sparkles } from 'lucide-react'

export function UnlockCard({ title, text, cta, onClick, icon: Icon = Sparkles }) {
  return (
    <section style={{ marginBottom: 32 }}>
      <div style={{
        background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border)', padding: '28px 24px', textAlign: 'center',
      }}>
        <div style={{
          width: 44, height: 44, borderRadius: '50%', margin: '0 auto 14px',
          background: 'var(--accent-soft)', border: '1.5px solid var(--accent)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Icon size={18} color="var(--accent)" strokeWidth={1.5} />
        </div>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 15, fontWeight: 500,
          color: 'var(--text)', marginBottom: 8 }}>{title}</h3>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.6,
          maxWidth: 380, margin: '0 auto 18px' }}>{text}</p>
        <button className="analyse-btn" onClick={onClick}
          style={{ maxWidth: 240, margin: '0 auto' }}>{cta}</button>
      </div>
    </section>
  )
}