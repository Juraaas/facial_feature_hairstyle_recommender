import { useState } from 'react'
import { analysePhoto } from '../api/client'

export function useAnalysis() {
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  async function analyse(file) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await analysePhoto(file)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function reset() {
    setResult(null)
    setError(null)
  }

  return { result, loading, error, analyse, reset }
}