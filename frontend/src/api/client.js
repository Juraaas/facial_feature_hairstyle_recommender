const BASE = import.meta.env.VITE_API_URL || '/api'

export async function analysePhoto(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/analyse`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Analysis failed')
  }
  return res.json()
}

export async function sendVote(styleName, vote, features, gender) {
  await fetch(`${BASE}/vote`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ style_name: styleName, vote, features, gender }),
  })
}

export async function sendFeedback(features, qualityScore, topStyles, rating, comment) {
  await fetch(`${BASE}/feedback`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      features,
      quality_score: qualityScore,
      top_styles:    topStyles,
      rating,
      comment,
    }),
  })
}