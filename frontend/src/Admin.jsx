import { useEffect, useState } from 'react'
import { supabase } from './lib/supabase'
import { useAuth } from './hooks/useAuth'

const ADMIN_EMAIL = 'pozdroelobenc@gmail.com'

export function Admin() {
  const { user } = useAuth()
  const [stats, setStats] = useState(null)
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      const [
        { count: totalUsers },
        { count: totalAnalyses },
        { count: premiumUsers },
        { data: recentUsers },
        { data: ratings },
      ] = await Promise.all([
        supabase.from('profiles').select('*', { count: 'exact', head: true }),
        supabase.from('analyses').select('*', { count: 'exact', head: true }),
        supabase.from('profiles').select('*', { count: 'exact', head: true })
          .eq('plan', 'premium'),
        supabase.from('profiles').select('id, email, plan, created_at')
          .order('created_at', { ascending: false }).limit(20),
        supabase.from('preview_ratings').select('style_name, rating'),
      ])
      const ratingStats = (ratings || []).reduce((acc, r) => {
        if (!acc[r.style_name]) acc[r.style_name] = { good: 0, bad: 0, total: 0 }
        acc[r.style_name][r.rating]++
        acc[r.style_name].total++
        return acc
      }, {})

      setStats({ totalUsers, totalAnalyses, premiumUsers })
      setUsers(recentUsers || [])
      setLoading(false)
    }
    load()
  }, [user])

  async function upgradeToPremium(userId) {
    await supabase.from('profiles').update({ plan: 'premium' }).eq('id', userId)
    setUsers(prev => prev.map(u => u.id === userId ? { ...u, plan: 'premium' } : u))
  }

  async function downgradeToFree(userId) {
    await supabase.from('profiles').update({ plan: 'free' }).eq('id', userId)
    setUsers(prev => prev.map(u => u.id === userId ? { ...u, plan: 'free' } : u))
  }

  if (!user) return <div style={{ padding: 40 }}>Not logged in</div>
  if (user.email !== ADMIN_EMAIL) return <div style={{ padding: 40 }}>Access denied: {user.email}</div>
  if (loading || !stats) return <div style={{ padding: 40 }}>Loading...</div>

  return (
    <div style={{ padding: 40, maxWidth: 900, margin: '0 auto',
      fontFamily: 'var(--font-body)', color: 'var(--text)' }}>
      <h1 style={{ fontFamily: 'var(--font-display)', marginBottom: 32 }}>
        Admin Panel
      </h1>

      {/* stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
        gap: 16, marginBottom: 40 }}>
        {[
          { label: 'Total users', value: stats.totalUsers },
          { label: 'Premium users', value: stats.premiumUsers },
          { label: 'Total analyses', value: stats.totalAnalyses },
        ].map(s => (
          <div key={s.label} style={{
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', padding: '20px', textAlign: 'center',
          }}>
            <p style={{ fontSize: 32, fontWeight: 600,
              fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
              {s.value}
            </p>
            <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>{s.label}</p>
          </div>
        ))}
      </div>

      {/* users table */}
      <h2 style={{ marginBottom: 16, fontSize: 16 }}>Recent users</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: '2px solid var(--border)' }}>
            {['Email', 'Plan', 'Joined', 'Actions'].map(h => (
              <th key={h} style={{ padding: '8px 12px', textAlign: 'left',
                color: 'var(--text-hint)', fontWeight: 500 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {users.map(u => (
            <tr key={u.id} style={{ borderBottom: '1px solid var(--border)' }}>
              <td style={{ padding: '8px 12px' }}>{u.email}</td>
              <td style={{ padding: '8px 12px' }}>
                <span style={{
                  fontSize: 10, padding: '2px 8px', borderRadius: 20,
                  background: u.plan === 'premium' ? 'var(--accent-soft)' : 'var(--surface-2)',
                  color: u.plan === 'premium' ? 'var(--accent)' : 'var(--text-muted)',
                  border: `1px solid ${u.plan === 'premium' ? 'var(--accent)' : 'var(--border)'}`,
                }}>
                  {u.plan}
                </span>
              </td>
              <td style={{ padding: '8px 12px', color: 'var(--text-muted)', fontSize: 11 }}>
                {new Date(u.created_at).toLocaleDateString()}
              </td>
              <td style={{ padding: '8px 12px' }}>
                {u.plan === 'free' ? (
                  <button onClick={() => upgradeToPremium(u.id)} style={{
                    fontSize: 11, padding: '3px 10px', borderRadius: 20,
                    background: 'var(--accent)', color: '#fff',
                    border: 'none', cursor: 'pointer',
                  }}>→ Premium</button>
                ) : (
                  <button onClick={() => downgradeToFree(u.id)} style={{
                    fontSize: 11, padding: '3px 10px', borderRadius: 20,
                    background: 'none', color: 'var(--text-muted)',
                    border: '1px solid var(--border)', cursor: 'pointer',
                  }}>→ Free</button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* preview ratings */}
        {stats.ratingStats && Object.keys(stats.ratingStats).length > 0 && (
          <>
            <h2 style={{ margin: '40px 0 16px', fontSize: 16 }}>Preview ratings by style</h2>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--border)' }}>
                  {['Style', 'Good', 'Bad', 'Total', 'Score'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left',
                      color: 'var(--text-hint)', fontWeight: 500 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(stats.ratingStats)
                  .sort((a, b) => b[1].total - a[1].total)
                  .map(([style, r]) => (
                  <tr key={style} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '8px 12px' }}>{style}</td>
                    <td style={{ padding: '8px 12px', color: '#2d8f4e' }}>{r.good}</td>
                    <td style={{ padding: '8px 12px', color: '#c0392b' }}>{r.bad}</td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }}>{r.total}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{
                        fontSize: 11, padding: '2px 8px', borderRadius: 20,
                        background: r.good / r.total > 0.7 ? '#f0faf4' : '#fef4f2',
                        color: r.good / r.total > 0.7 ? '#2d8f4e' : '#c0392b',
                        border: `1px solid ${r.good / r.total > 0.7 ? '#b7dfc7' : '#f5c6bc'}`,
                      }}>
                        {Math.round(r.good / r.total * 100)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
    </div>
  )
}