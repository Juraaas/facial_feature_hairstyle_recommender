import { Lock } from 'lucide-react'

export function LockedBlock({ title, text, lines = 3, onClick }) {
  return (
    <section style={{ marginBottom: 32 }}>
      <h2 className="section-title" style={{ marginBottom: 12 }}>{title}</h2>
      <button type="button" onClick={onClick} style={{
        position: 'relative', display: 'block', width: '100%', textAlign: 'left', cursor: 'pointer',
        background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)', padding: '20px', overflow: 'hidden',
      }}>
        <div aria-hidden="true" style={{
          display: 'flex', flexDirection: 'column', gap: 10, filter: 'blur(3px)', opacity: 0.5,
        }}>
          {Array.from({ length: lines }).map((_, i) => (
            <div key={i} style={{
              height: 10, borderRadius: 5, background: 'var(--border)', width: `${94 - i * 14}%`,
            }} />
          ))}
        </div>
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 6, padding: '6px 14px', borderRadius: 20,
            background: 'var(--surface)', border: '1px solid var(--accent)',
            color: 'var(--accent)', fontSize: 12, fontWeight: 500,
          }}>
            <Lock size={13} strokeWidth={1.5} />{text}
          </span>
        </div>
      </button>
    </section>
  )
}