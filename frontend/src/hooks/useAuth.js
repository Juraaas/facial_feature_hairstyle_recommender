import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'

export function useAuth() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [recovery, setRecovery] = useState(() => sessionStorage.getItem('pw_recovery') === '1')

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setUser(data.session?.user ?? null)
      setLoading(false)
    })

    const { data: listener } = supabase.auth.onAuthStateChange((event, session) => {
      setUser(session?.user ?? null)
      if (event === 'PASSWORD_RECOVERY') {
        sessionStorage.setItem('pw_recovery', '1')
        setRecovery(true)
      }
      if (event === 'SIGNED_OUT') {
        sessionStorage.removeItem('pw_recovery')
        setRecovery(false)
      }
    })

    return () => listener.subscription.unsubscribe()
  }, [])

  function clearRecovery() {
    sessionStorage.removeItem('pw_recovery')
    setRecovery(false)
  }

  async function signOut() {
    await supabase.auth.signOut()
  }

  async function getToken() {
    const { data } = await supabase.auth.getSession()
    return data.session?.access_token ?? null
  }

  return { user, loading, recovery, clearRecovery, signOut, getToken }
}