import { supabase } from '../lib/supabase'

const BASE = import.meta.env.VITE_API_URL || '/api'

async function getAuthHeaders() {
  const { data } = await supabase.auth.getSession()
  const token = data.session?.access_token
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function analysePhoto(file, lang) {
  const form = new FormData()
  form.append('file', file)
  const headers = await getAuthHeaders()
  const res = await fetch(`${BASE}/analyse?lang=${lang}`, 
    { method: 'POST', body: form, headers})
  if (!res.ok) {
    const err = await res.json()
    throw new Error(JSON.stringify(
            err.detail || { code: "INTERNAL_ERROR", message: "Analysis failed" }
        ))
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

export async function createCheckout() {
  const headers = await getAuthHeaders()
  const res = await fetch(`${BASE}/create-checkout`, {
    method: 'POST', headers,
  })
  if (!res.ok) throw new Error('Checkout failed')
  const { url } = await res.json()
  window.location.href = url 
}