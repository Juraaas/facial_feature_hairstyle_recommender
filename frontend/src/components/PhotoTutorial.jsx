import { useState } from "react"

const STEPS = [
    {
        title: "Position your face",
        items: [
            { ok: true, text: "Face the camera directly - look straight at the lens"},
            { ok: true, text: "Keep your head uprgith and level"},
            { ok: true, text: "Fill at least half of the frame with your face"},
            { ok: false, text: "Don't tilt or turn your head to the side"},
            { ok: false, text: "Don't take the photo from above or below"},
        ],
        visual: (
            <div style={{ position: "relative", width: 120, height: 120, margin: '0 auto'}}>
                <div style={{ 
                    width: 120, height: 120, borderRadius: '50%', background: 'var(--bg)',
                    border: '2px solid var(--accent-mid)', display: 'flex', alignItems: 'center',
                    justifyContent: 'center', fontSize: 56
                }}>🧑</div>
                <div style={{
                    position: 'absolute', bottom: 0, right: 0, background: '#2d8f4e',
                    borderRadius: '50%', width: 28, height: 28, display: 'flex',
                    alignItems: 'center', justifyContent: 'center', fontSize: 14
                }}></div>
            </div>
        ),
    },
    {
        title: "Lighting & background",
        items: [
            { ok: true, text: "Use natural light or face a lamp - avoid backlighting"},
            { ok: true, text: "Plain or neutral background works best"},
            { ok: true, text: "Make sure your face is evenly lit on both sides"},
            { ok: true, text: "Avoid strong shadows across your face"},
            { ok: true, text: "Avoid very dark or very bright environments"},   
        ],
        visual: (
        <div style={{ width: 120, height: 120, margin: '0 auto', borderRadius: 12,
            background: 'var(--bg)', border: '2px solid var(--accent-mid)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 48
        }}>💡</div>
        ),
    },
    {
        title: "Hair & accessories",
        items: [
            { ok: true, text: "Keep hair away from your face and forehead"},
            { ok: true, text: "Show your natural hairline clearly"},
            { ok: true, text: "Remove hats, hoods, and headbands"},
            { ok: true, text: "Don't wear sunglasses or large glasses if possible"},
            { ok: true, text: "Don't cover your forehead with a fringe or hat"},   
        ],
        visual: (
        <div style={{ width: 120, height: 120, margin: '0 auto', borderRadius: 12,
            background: 'var(--bg)', border: '2px solid var(--accent-mid)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 48
        }}>✂️</div>
        ),
    },
]

export function PhotoTutorial({ onDone }) {
  const [step, setStep] = useState(0)
  const current = STEPS[step]
  const isLast  = step === STEPS.length - 1

  return (
    <div style={{
      background:   'var(--surface)',
      border:       '0.5px solid var(--border)',
      borderRadius: 16,
      padding:      '28px 24px',
      marginBottom: 24,
      animation:    'fadeIn .3s ease',
    }}>
      {/* progress dots */}
      <div style={{ display: 'flex', gap: 6, justifyContent: 'center', marginBottom: 24 }}>
        {STEPS.map((_, i) => (
          <div key={i} style={{
            width: i === step ? 20 : 6, height: 6,
            borderRadius: 3,
            background: i === step ? 'var(--accent-mid)'
                      : i < step  ? 'var(--accent-mid)'
                      : 'var(--border)',
            transition: 'all .3s',
            opacity: i < step ? 0.4 : 1,
          }} />
        ))}
      </div>

      {/* visual */}
      <div style={{ marginBottom: 20 }}>
        {current.visual}
      </div>

      {/* title */}
      <h3 style={{
        fontSize: 16, fontWeight: 600,
        color: 'var(--text)', textAlign: 'center',
        marginBottom: 16,
      }}>
        {current.title}
      </h3>

      {/* items */}
      <div style={{ marginBottom: 24 }}>
        {current.items.map((item, i) => (
          <div key={i} style={{
            display: 'flex', alignItems: 'flex-start', gap: 10,
            marginBottom: 10,
          }}>
            <span style={{
              fontSize: 13, flexShrink: 0, marginTop: 1,
              color: item.ok ? '#2d8f4e' : '#c0392b',
            }}>
              {item.ok ? '✓' : '✗'}
            </span>
            <span style={{
              fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.5,
            }}>
              {item.text}
            </span>
          </div>
        ))}
      </div>

      {/* navigation */}
      <div style={{ display: 'flex', gap: 8 }}>
        {step > 0 && (
          <button
            onClick={() => setStep(s => s - 1)}
            style={{
              flex: 1, padding: '10px 0',
              background: 'none',
              border: '0.5px solid var(--border)',
              borderRadius: 8, cursor: 'pointer',
              fontSize: 13, color: 'var(--text-muted)',
            }}
          >
            Back
          </button>
        )}
        <button
          onClick={() => isLast ? onDone() : setStep(s => s + 1)}
          style={{
            flex: 2, padding: '10px 0',
            background: 'var(--accent)',
            border: 'none', borderRadius: 8,
            cursor: 'pointer', fontSize: 13,
            color: '#fff', fontWeight: 500,
          }}
        >
          {isLast ? "Got it — take my photo" : "Next"}
        </button>
      </div>

      {/* skip */}
      <button
        onClick={onDone}
        style={{
          display: 'block', width: '100%',
          marginTop: 10, padding: '6px 0',
          background: 'none', border: 'none',
          cursor: 'pointer', fontSize: 12,
          color: 'var(--text-hint)',
        }}
      >
        Skip tutorial
      </button>
    </div>
  )
}