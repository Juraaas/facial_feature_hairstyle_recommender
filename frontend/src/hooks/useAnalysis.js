import { useState } from 'react'
import { analysePhoto, ensureServer, parseError } from '../api/client'

export function useAnalysis() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [warming, setWarming] = useState(false)
  const [error, setError] = useState(null)

  async function analyse(file, lang, opts = {}) {
    setLoading(true); setError(null); setResult(null)
    try {
      const up = await ensureServer(() => setWarming(true))
      setWarming(false)
      if (!up) {
        setError({ code: 'SERVER_UNAVAILABLE', message: 'Health check did not succeed' })
        return null
      }
      const data = await analysePhoto(file, lang, opts)
      setResult(data)
      return data
    } catch (e) {
      console.error('analyse failed:', e)
      setError(parseError(e))
      return null
    } finally {
      setLoading(false); setWarming(false)
    }
  }

  function reset() { setResult(null); setError(null) }

  return { result, loading, warming, error, analyse, reset }
}