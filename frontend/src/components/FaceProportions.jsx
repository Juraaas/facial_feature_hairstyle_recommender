import { TraitBar } from "./TraitBar"
import { useTranslation } from 'react-i18next'

const BARS = [
  ['face_ratio',       'trait_face_shape',     'label_wide_face',     'label_long_face'],
  ['jaw_ratio',        'trait_jaw_width',      'label_narrow_jaw',    'label_wide_jaw'],
  ['cheekbone_to_jaw', 'trait_cheekbone',      'label_jaw_dominant',  'label_cheek_dominant'],
  ['eye_ratio',        'trait_eye_spacing',    'label_close_eyes',    'label_wide_eyes'],
  ['lip_ratio',        'trait_lip_width',      'label_narrow_lips',   'label_wide_lips'],
  ['nose_position',    'trait_nose_position',  'label_low_nose',      'label_high_nose'],
  ['lower_face_ratio', 'trait_lower_face',     'label_short_lower',   'label_long_lower'],
  ['chin_prominence',  'trait_chin',           'label_flat_chin',     'label_strong_chin'],
  ['symmetry',         'trait_symmetry',       'label_symmetrical',   'label_asymmetrical'],
  ['upper_third',      'trait_forehead',       'label_low_forehead',  'label_high_forehead'],
]

const SYMMETRY_LABELS = {
  pl: { high: 'symetryczna', medium: 'umiarkowana', low: 'asymetryczna' },
  en: { high: 'symmetrical', medium: 'moderate',    low: 'asymmetrical' },
}

const BALANCE_LABELS = {
  pl: { balanced: 'Zbalansowane', slight_imbalance: 'Lekka dysproporcja', imbalanced: 'Dysproporcja' },
  en: { balanced: 'Balanced',     slight_imbalance: 'Slight imbalance',   imbalanced: 'Imbalanced' },
}

function ThirdsBalance({ features, traits, pl }) {
  const u = features.upper_third
  const m = features.middle_third
  const l = features.lower_third
  if (u === undefined || m === undefined || l === undefined) return null

  // te same progi co w face_traits.py
  const maxDev = Math.max(Math.abs(u - 0.333), Math.abs(m - 0.333), Math.abs(l - 0.333))
  const computed = maxDev < 0.05 ? 'balanced' : maxDev < 0.10 ? 'slight_imbalance' : 'imbalanced'
  const state = traits?.thirds_balance ?? computed
  const L = BALANCE_LABELS[pl ? 'pl' : 'en']

  const ok = state === 'balanced'
  const color = ok ? '#2d8f4e' : state === 'slight_imbalance' ? '#C8975A' : '#c0392b'

  const parts = [
    { v: u, label: pl ? 'Czoło' : 'Forehead' },
    { v: m, label: pl ? 'Środek' : 'Middle' },
    { v: l, label: pl ? 'Dół'   : 'Lower' },
  ]

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginBottom: 8,
      }}>
        <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text)' }}>
          {pl ? 'Proporcje tercji twarzy' : 'Facial thirds balance'}
        </span>
        <span style={{
          fontSize: 10, padding: '2px 10px', borderRadius: 20, fontWeight: 500,
          color, border: `1px solid ${color}`, background: 'var(--surface)',
        }}>
          {L[state]}
        </span>
      </div>

      <div style={{ display: 'flex', gap: 3, height: 26 }}>
        {parts.map(p => (
          <div key={p.label} style={{
            flex: p.v, borderRadius: 6, minWidth: 0,
            background: 'var(--accent-soft)', border: '1px solid var(--border)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)',
          }}>
            {Math.round(p.v * 100)}%
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 3, marginTop: 5 }}>
        {parts.map(p => (
          <span key={p.label} style={{
            flex: p.v, textAlign: 'center', fontSize: 10,
            color: 'var(--text-hint)', fontFamily: 'var(--font-mono)',
          }}>
            {p.label}
          </span>
        ))}
      </div>

      <p style={{
        fontSize: 10, color: 'var(--text-hint)', marginTop: 6,
        textAlign: 'center', fontWeight: 300,
      }}>
        {pl ? 'Idealnie każda część ≈ 33%' : 'Ideal is each part ≈ 33%'}
      </p>
    </div>
  )
}

export function FaceProportions({ features, norms, traits }) {
  const { t, i18n } = useTranslation()
  const pl = i18n.language === 'pl'

  return (
    <section style={{ marginBottom: 32 }}>
      <h2 className="section-title">{t('section_proportions')}</h2>
      <div style={{
        background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border)', padding: '20px 20px 8px',
      }}>
        {BARS.map(([feat, title, minLabel, maxLabel]) => {
          const n = norms[feat]
          if (!n || features[feat] === undefined) return null

          const isSym = feat === 'symmetry'
          return (
            <TraitBar
              key={feat}
              title={t(title)}
              value={features[feat]}
              minVal={n.p5}
              maxVal={n.p95}
              avgVal={n.mean}
              minLabel={t(minLabel)}
              maxLabel={t(maxLabel)}
              scale={isSym ? 'sqrt' : 'linear'}
              interp={isSym && traits?.symmetry
                ? SYMMETRY_LABELS[pl ? 'pl' : 'en'][traits.symmetry]
                : undefined}
            />
          )
        })}
        <ThirdsBalance features={features} traits={traits} pl={pl} />
      </div>
    </section>
  )
}