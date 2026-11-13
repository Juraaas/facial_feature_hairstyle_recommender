const TRAITS = {
  face_shape_type: { label: ['Kształt twarzy', 'Face shape'], values: {
    oval: ['owalna', 'oval'],
    triangle: ['trójkątna (szersze kości policzkowe)', 'triangle (wider cheekbones)'],
    square: ['kwadratowa (mocna szczęka)', 'square (strong jaw)'] } },
  face_length: { label: ['Długość twarzy', 'Face length'], values: {
    long: ['wydłużona', 'long'], short: ['krótka', 'short'], balanced: ['zrównoważona', 'balanced'] } },
  jaw: { label: ['Szczęka', 'Jaw'], values: {
    narrow: ['wąska', 'narrow'], wide: ['szeroka', 'wide'], normal: ['średnia', 'average'] } },
  jaw_height: { label: ['Wysokość szczęki', 'Jaw height'], values: {
    high: ['wysoka', 'high'], low: ['niska', 'low'], normal: ['średnia', 'average'] } },
  chin: { label: ['Broda', 'Chin'], values: {
    prominent: ['wyraźna', 'prominent'], recessed: ['cofnięta', 'recessed'], normal: ['proporcjonalna', 'average'] } },
  lower_face: { label: ['Dolna część twarzy', 'Lower face'], values: {
    long: ['długa', 'long'], short: ['krótka', 'short'], normal: ['proporcjonalna', 'average'] } },
  forehead: { label: ['Czoło', 'Forehead'], values: {
    high: ['wysokie', 'high'], low: ['niskie', 'low'], normal: ['średnie', 'average'] } },
  eyes: { label: ['Rozstaw oczu', 'Eye spacing'], values: {
    wide: ['szeroki', 'wide-set'], close: ['wąski', 'close-set'], normal: ['średni', 'average'] } },
  lips: { label: ['Usta', 'Lips'], values: {
    wide: ['szerokie', 'wide'], narrow: ['wąskie', 'narrow'], normal: ['średnie', 'average'] } },
  nose: { label: ['Pozycja nosa', 'Nose position'], values: {
    'upper-dominant': ['osadzony wysoko', 'set high'], 'lower-dominant': ['osadzony nisko', 'set low'],
    balanced: ['pośrodku', 'centered'] } },
  symmetry: { label: ['Symetria', 'Symmetry'], values: {
    high: ['wysoka', 'high'], medium: ['umiarkowana', 'moderate'], low: ['niska', 'low'] } },
  dominant_third: { label: ['Dominująca część twarzy', 'Dominant third'], values: {
    upper: ['górna', 'upper'], middle: ['środkowa', 'middle'], lower: ['dolna', 'lower'],
    balanced: ['brak dominacji', 'none'] } },
  thirds_vertical: { label: ['Rozkład pionowy', 'Vertical balance'], values: {
    top_heavy: ['dominuje góra', 'top-heavy'], bottom_heavy: ['dominuje dół', 'bottom-heavy'],
    balanced: ['zrównoważony', 'balanced'] } },
  thirds_balance: { label: ['Balans proporcji', 'Proportion balance'], values: {
    balanced: ['zrównoważony', 'balanced'], slight_imbalance: ['lekka dysproporcja', 'slight imbalance'],
    imbalanced: ['dysproporcja', 'imbalanced'] } },
  hair_type: { label: ['Typ włosów', 'Hair type'], values: {
    straight: ['proste', 'straight'], wavy: ['falowane', 'wavy'], curly: ['kręcone', 'curly'],
    coily: ['mocno skręcone (coily)', 'coily'] } },
  hairline: { label: ['Linia włosów', 'Hairline'], values: {
    normal: ['prawidłowa', 'normal'], receding: ['cofnięta', 'receding'], uneven: ['nieregularna', 'uneven'] } },
}

export const TRAIT_GROUPS = [
  { id: 'face', pl: 'Twarz', en: 'Face',
    keys: ['face_shape_type', 'face_length', 'jaw', 'jaw_height', 'chin', 'lower_face', 'forehead'] },
  { id: 'features', pl: 'Rysy', en: 'Features', keys: ['eyes', 'lips', 'nose'] },
  { id: 'proportions', pl: 'Proporcje', en: 'Proportions',
    keys: ['symmetry', 'dominant_third', 'thirds_vertical', 'thirds_balance'] },
  { id: 'hair', pl: 'Włosy', en: 'Hair', keys: ['hair_type', 'hairline'] },
]

export function traitLabel(key, pl) {
  const t = TRAITS[key]
  return t ? t.label[pl ? 0 : 1] : key.replace(/_/g, ' ')
}

export function traitValue(key, value, pl) {
  const v = TRAITS[key]?.values?.[String(value)]
  return v ? v[pl ? 0 : 1] : String(value).replace(/[_-]/g, ' ')
}