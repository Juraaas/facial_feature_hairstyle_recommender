import { useState, useEffect, useRef } from 'react'
import { supabase } from '../lib/supabase'

const REVOKED = ['session_not_found', 'user_not_found', 'bad_jwt',
                 'refresh_token_not_found', 'refresh_token_already_used']

export function useAuth() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [recovery, setRecovery] = useState(() => sessionStorage.getItem('pw_recovery') === '1')
  const lastCheck = useRef(0)

  useEffect(() => {
    let cancelled = false

    async function verifySession() {
      const now = Date.now()
      if (now - lastCheck.current < 15000) return
      lastCheck.current = now
      const { data } = await supabase.auth.getSession()
      if (!data.session) return
      const { error } = await supabase.auth.getUser()
      if (error && !cancelled) {
        const revoked = error.status === 401 || error.status === 403 || REVOKED.includes(error.code)
        if (revoked) await supabase.auth.signOut({ scope: 'local' }).catch(() => {})
      }
    }

    supabase.auth.getSession().then(({ data }) => {
      setUser(data.session?.user ?? null)
      setLoading(false)
      verifySession()
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

    const onVisible = () => { if (document.visibilityState === 'visible') verifySession() }
    document.addEventListener('visibilitychange', onVisible)
    window.addEventListener('focus', verifySession)

    return () => {
      cancelled = true
      listener.subscription.unsubscribe()
      document.removeEventListener('visibilitychange', onVisible)
      window.removeEventListener('focus', verifySession)
    }
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