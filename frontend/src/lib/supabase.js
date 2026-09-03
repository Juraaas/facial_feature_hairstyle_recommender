import { createClient } from '@supabase/supabase-js'

export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY
)

async function signUpWithEmail(email, password) {
  const { error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo: 'https://stylizzer.vercel.app/analyse',
    }
  })
  if (error) throw error
}