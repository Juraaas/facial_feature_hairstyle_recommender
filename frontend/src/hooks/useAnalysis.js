import { useState } from 'react'
import { analysePhoto } from '../api/client'

export function useAnalysis() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function analyse(file, lang, opts = {}) {
    setLoading(true); setError(null); setResult(null)
    try { 
      const data = await analysePhoto(file, lang, opts)
      setResult(data)
      return data
    } catch (e) {
      try { setError(JSON.parse(e.message)) } 
      catch { setError({ code: "INTERNAL_ERROR", message: e.message }) }
      return null
    } finally {
      setLoading(false)
    }
  }

  function reset() { setResult(null); setError(null) }

  return { result, loading, error, analyse, reset }
}