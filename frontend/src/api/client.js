import { supabase } from '../lib/supabase'

export const API_BASE = import.meta.env.VITE_API_URL
const BASE = API_BASE

async function getAuthHeaders() {
  const { data } = await supabase.auth.getSession()
  const token = data.session?.access_token
  return token ? { Authorization: `Bearer ${token}` } : {}
}

const isJson = res => (res.headers.get('content-type') || '').includes('application/json')
const fail = (code, message) => new Error(JSON.stringify({ code, message }))

async function safeFetch(url, options) {
  try {
    return await fetch(url, options)
  } catch (e) {
    console.error('Network error', url, e)
    throw fail('SERVER_UNAVAILABLE', `Network error: ${e.message}`)
  }
}

async function readJson(res) {
  if (!isJson(res)) {
    console.error('Non-JSON response', res.status, res.url)
    throw fail('SERVER_UNAVAILABLE', `Non-JSON response (HTTP ${res.status})`)
  }
  return res.json()
}

async function apiError(res) {
  const err = isJson(res) ? await res.json().catch(() => null) : null
  let payload
  if (err?.detail && !Array.isArray(err.detail)) payload = err.detail
  else if (Array.isArray(err?.detail)) payload = { code: 'INVALID_INPUT', message: 'Invalid request' }
  else if (err?.code) payload = err
  else payload = {
    code: res.status === 429 ? 'RATE_LIMITED' : 'SERVER_UNAVAILABLE',
    message: `HTTP ${res.status}`,
  }
  console.error('API error', res.status, payload)
  return new Error(JSON.stringify(payload))
}

export function parseError(e) {
  try {
    const p = JSON.parse(e.message)
    if (p && p.code) return p
  } catch { /* nie JSON */ }
  return { code: 'INTERNAL_ERROR', message: e?.message || 'Unknown error' }
}

let serverReady = false
export async function ensureServer(onWarming, { tries = 25, delay = 4000 } = {}) {
  if (serverReady) return true
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(`${API_BASE}/health`, { cache: 'no-store' })
      if (res.ok && isJson(res)) { serverReady = true; return true }
    } catch { /* serwer śpi albo brak CORS */ }
    if (i === 0) onWarming?.()
    await new Promise(r => setTimeout(r, delay))
  }
  return false
}

export async function analysePhoto(file, lang, { gender = null, full = false, analysisId = null } = {}) {
  const form = new FormData()
  form.append('file', file)
  const headers = await getAuthHeaders()
  const params = new URLSearchParams({ lang })
  if (gender) params.append('gender', gender)
  if (full) params.append('full', 'true')
  if (analysisId) params.append('analysis_id', analysisId)
  const res = await safeFetch(`${BASE}/analyse?${params}`, { method: 'POST', body: form, headers })
  if (!res.ok) throw await apiError(res)
  return readJson(res)
}

export async function fetchMe() {
  const headers = await getAuthHeaders()
  const res = await safeFetch(`${BASE}/me`, { headers })
  if (!res.ok) throw await apiError(res)
  return readJson(res)
}

export async function createCheckout(packId) {
  const headers = { ...(await getAuthHeaders()), 'Content-Type': 'application/json' }
  const res = await safeFetch(`${BASE}/create-checkout`, {
    method: 'POST', headers, body: JSON.stringify({ pack_id: packId }),
  })
  if (!res.ok) throw await apiError(res)
  const { url } = await readJson(res)
  window.location.href = url
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
      features, quality_score: qualityScore, top_styles: topStyles, rating, comment,
    }),
  })
}