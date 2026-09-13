import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ShieldCheck, User } from 'lucide-react'


export function ConsentCheckboxes({ onChange }) {
  const { i18n } = useTranslation()
  const pl = i18n.language === 'pl'

  const [photoConsent, setPhotoConsent] = useState(false)
  const [ageConsent, setAgeConsent]   = useState(false)

  function handlePhoto(checked) {
    setPhotoConsent(checked)
    onChange(checked && ageConsent)
  }

  function handleAge(checked) {
    setAgeConsent(checked)
    onChange(photoConsent && checked)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 12 }}>
      <CheckRow
        checked={photoConsent}
        onChange={handlePhoto}
        Icon={ShieldCheck}
      >
        {pl ? (
          <>
            Wyrażam zgodę na jednorazowe przetwarzanie mojego zdjęcia twarzy
            w celu analizy geometrycznej. Zdjęcie nie jest przechowywane po zakończeniu analizy.{' '}
            <a href="/privacy" target="_blank" rel="noopener noreferrer"
              style={{ color: 'var(--accent)', textDecoration: 'none' }}
              onClick={e => e.stopPropagation()}>
              Polityka Prywatności
            </a>
          </>
        ) : (
          <>
            I consent to one-time processing of my facial photo for geometric analysis.
            The photo is not stored after the analysis is complete.{' '}
            <a href="/privacy" target="_blank" rel="noopener noreferrer"
              style={{ color: 'var(--accent)', textDecoration: 'none' }}
              onClick={e => e.stopPropagation()}>
              Privacy Policy
            </a>
          </>
        )}
      </CheckRow>

      <CheckRow
        checked={ageConsent}
        onChange={handleAge}
        Icon={User}
      >
        {pl
          ? 'Potwierdzam, że mam ukończone 16 lat.'
          : 'I confirm that I am 16 years of age or older.'}
      </CheckRow>
    </div>
  )
}

function CheckRow({ checked, onChange, Icon, children }) {
  return (
    <label style={{
      display: 'flex', alignItems: 'flex-start', gap: 10, cursor: 'pointer', userSelect: 'none',
    }}>
      {/* custom checkbox */}
      <div
        onClick={() => onChange(!checked)}
        style={{
          width: 18, height: 18, borderRadius: 4,
          border: `1.5px solid ${checked ? 'var(--accent)' : 'var(--border)'}`,
          background: checked ? 'var(--accent)' : 'var(--surface-2)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0, marginTop: 1, transition: 'all .15s',
        }}
      >
        {checked && (
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M1.5 5L4 7.5L8.5 2.5" stroke="#fff" strokeWidth="1.5"
              strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6, flex: 1 }}>
        <Icon size={13} strokeWidth={1.5} color="var(--text-hint)"
          style={{ marginTop: 2, flexShrink: 0 }} />
        <span style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6, fontWeight: 300 }}>
          {children}
        </span>
      </div>
    </label>
  )
}