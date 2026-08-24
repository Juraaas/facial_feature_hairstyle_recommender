import { TraitBar } from "./TraitBar"
import { useTranslation } from 'react-i18next'

const BARS = [
  ['face_ratio',       'trait_face_shape',   'label_wide_face',     'label_long_face'],
  ['jaw_ratio',        'trait_jaw_width',     'label_narrow_jaw',    'label_wide_jaw'],
  ['eye_ratio',        'trait_eye_spacing',   'label_close_eyes',    'label_wide_eyes'],
  ['eye_height',       'trait_eye_openness',  'label_narrow_eyes',   'label_wide_eyes2'],
  ['lip_ratio',        'trait_lip_width',     'label_narrow_lips',   'label_wide_lips'],
  ['nose_position',    'trait_nose_position', 'label_low_nose',      'label_high_nose'],
  ['lower_face_ratio', 'trait_lower_face',    'label_short_lower',   'label_long_lower'],
  ['chin_prominence',  'trait_chin',          'label_flat_chin',     'label_strong_chin'],
  ['symmetry',         'trait_symmetry',      'label_symmetrical',   'label_asymmetrical'],
  ['upper_third',      'trait_forehead',      'label_low_forehead',  'label_high_forehead'],
  ['middle_third',     'trait_mid_face',      'label_short_mid',     'label_long_mid'],
  ['mid_lower_ratio',  'trait_mid_lower',     'label_lower_dom',     'label_mid_dom'],
]

export function FaceProportions({ features, norms }) {
    const { t } = useTranslation()

    return (
        <section style={{ marginBottom: 32 }}>
            <h2 className="section-title">{t('section_proportions')}</h2>
            <div style={{background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border)', padding: '20px 20px 8px'}}>
                {BARS.map(([feat, title, minLabel, maxLabel]) => {
                    const n = norms[feat]
                    if (!n || features[feat] === undefined) return null
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
                        />
                    )
                })}
            </div>
        </section>
    )
}