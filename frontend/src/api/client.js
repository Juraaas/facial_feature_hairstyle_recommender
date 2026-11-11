import { supabase } from '../lib/supabase'

export const API_BASE = import.meta.env.VITE_API_URL
const BASE = API_BASE

async function getAuthHeaders() {
  const { data } = await supabase.auth.getSession()
  const token = data.session?.access_token
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function apiError(res) {
  const err = await res.json().catch(() => null)
  let payload
  if (err?.detail && !Array.isArray(err.detail)) payload = err.detail
  else if (Array.isArray(err?.detail)) payload = { code: 'INVALID_INPUT', message: 'Invalid request' }
  else if (err?.code) payload = err
  else payload = {
    code: res.status === 429 ? 'RATE_LIMITED' : 'SERVER_UNAVAILABLE',
    message: 'Request failed',
  }
  return new Error(JSON.stringify(payload))
}

export async function analysePhoto(file, lang, { gender = null, analysisId = null } = {}) {
  const form = new FormData()
  form.append('file', file)
  const headers = await getAuthHeaders()
  const params = new URLSearchParams({ lang })
  if (gender) params.append('gender', gender)  
  if (full) params.append('full', 'true')
  if (analysisId) params.append('analysis_id', analysisId)
  const res = await fetch(`${BASE}/analyse?${params}`, 
    { method: 'POST', body: form, headers})
  if (!res.ok) throw await apiError(res)
  return res.json()
}

export async function createCheckout(packId) {
  const headers = { ...(await getAuthHeaders()), 'Content-Type': 'application/json' }
  const res = await fetch(`${BASE}/create-checkout`, {
    method: 'POST', headers, body: JSON.stringify({ pack_id: packId }),
  })
  if (!res.ok) throw await apiError(res)
  const { url } = await res.json()
  window.location.href = url 
}

export async function fetchMe() {
  const headers = await getAuthHeaders()
  const res = await fetch(`${BASE}/me`, { headers })
  if (!res.ok) throw await apiError(res)
  return res.json()
}

export async function sendVote(styleName, vote, features, gender) {
  await fetch(`${BASE}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ style_name: styleName, vote, features, gender }),
  })
}

export async function sendFeedback(features, qualityScore, topStyles, rating, comment) {
  await fetch(`${BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      features,
      quality_score: qualityScore,
      top_styles: topStyles,
      rating,
      comment,
    }),
  })
}