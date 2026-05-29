import { TraitBar } from "./TraitBar"

const BARS = [
  ['face_ratio',       'Face shape',        'Wide face',        'Long face'],
  ['jaw_ratio',        'Jaw width',         'Narrow jaw',       'Wide jaw'],
  ['eye_ratio',        'Eye spacing',       'Close-set eyes',   'Wide-set eyes'],
  ['eye_height',       'Eye openness',      'Narrow eyes',      'Wide eyes'],
  ['lip_ratio',        'Lip width',         'Narrow lips',      'Wide lips'],
  ['nose_position',    'Nose position',     'High nose',        'Low nose'],
  ['lower_face_ratio', 'Lower face length', 'Short lower face', 'Long lower face'],
  ['chin_prominence',  'Chin prominence',   'Flat chin',        'Strong chin'],
  ['symmetry',         'Facial symmetry',   'Symmetrical',      'Asymmetrical'],
  ['upper_third',      'Forehead',          'Low forehead',     'High forehead'],
  ['middle_third',     'Mid face',          'Short mid face',   'Long mid face'],
  ['lower_third',      'Lower face thirds', 'Short lower',      'Long lower'],
]

export function FaceProportions({ features, norms }) {
    return (
        <section style={{ marginBottom: 32}}>
            <h2 style={{ fontSize: 18, fontWeight: 500, marginBottom: 20}}>
                Facial Proportions
            </h2>
            {BARS.map(([feat, title, minLabel, maxLabel]) => {
                const n = norms[feat]
                if (!n || features [feat] === undefined) return null
                return (
                    <TraitBar
                        key={feat}
                        title={title}
                        value={features[feat]}
                        minVal={n.p5}
                        maxVal={n.p95}
                        avgVal={n.mean}
                        minLabel={minLabel}
                        maxLabel={maxLabel}
                    />
                )
            })}
        </section>
    )
}