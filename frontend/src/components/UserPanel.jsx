import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { supabase } from '../lib/supabase'
import { API_BASE, fetchTickets, exportAccountData, deleteAccount, parseError } from '../api/client'
import { creditWord } from '../lib/credits'
import { TRAIT_GROUPS, traitLabel, traitValue } from '../lib/traitLabels'
import { TicketModal } from './TicketModal'
import { X, Trash, Coins, History, WandSparkles, User, Download, LifeBuoy, KeyRound } from 'lucide-react'
import { btnAccent } from '../styles/shared'
import { SetPasswordModal } from './SetPasswordModal'

const COLORS = {
  natural: ['Naturalny', 'Natural'], blonde: ['Blond', 'Blonde'], dark: ['Ciemny brąz', 'Dark brown'],
  black: ['Czarny', 'Black'], auburn: ['Rudy', 'Auburn'],
}

const LEDGER = {
  signup_bonus: ['Bonus na start', 'Welcome bonus'],
  purchase: ['Zakup pakietu', 'Pack purchase'],
  spend_analysis: ['Pełna analiza', 'Full analysis'],
  spend_preview: ['Podgląd fryzury', 'Hairstyle preview'],
  refund_failed: ['Zwrot za nieudaną operację', 'Refund for a failed operation'],
  refund_ticket: ['Zwrot po zgłoszeniu', 'Refund after support request'],
  admin_grant: ['Kredyty od administratora', 'Credits from admin'],
  admin_revoke: ['Korekta salda', 'Balance adjustment'],
}

const GEN_STATUS = {
  succeeded: { l: ['Udany', 'Done'], c: '#2d8f4e', bg: '#f0faf4', bd: '#b7dfc7' },
  failed: { l: ['Nieudany', 'Failed'], c: '#c0392b', bg: '#fef4f2', bd: '#f5c6bc' },
  pending: { l: ['W toku', 'In progress'], c: '#C8975A', bg: '#fef8f0', bd: '#f0d5a8' },
  refunded: { l: ['Zwrócony', 'Refunded'], c: 'var(--text-muted)', bg: 'var(--surface-2)', bd: 'var(--border)' },
}

const TICKET_STATUS = {
  open: { l: ['Otwarte', 'Open'], c: '#C8975A', bg: '#fef8f0', bd: '#f0d5a8' },
  in_progress: { l: ['W trakcie', 'In progress'], c: '#C8975A', bg: '#fef8f0', bd: '#f0d5a8' },
  resolved: { l: ['Rozpatrzone', 'Resolved'], c: '#2d8f4e', bg: '#f0faf4', bd: '#b7dfc7' },
  rejected: { l: ['Odrzucone', 'Rejected'], c: 'var(--text-muted)', bg: 'var(--surface-2)', bd: 'var(--border)' },
}

const CATEGORY_LABEL = {
  preview_quality: ['Jakość podglądu', 'Preview quality'],
  technical: ['Problem techniczny', 'Technical problem'],
  billing: ['Płatności i kredyty', 'Payments and credits'],
  other: ['Inne', 'Other'],
}

function ledgerLabel(row, pl) {
  const base = (LEDGER[row.reason] || [row.reason, row.reason])[pl ? 0 : 1]
  if (row.reason === 'purchase') {
    const m = /pack=(\w+)/.exec(row.note || '')
    if (m) return `${base} ${m[1][0].toUpperCase()}${m[1].slice(1)}`
  }
  return base
}

const sectionLabel = {
  fontSize: 10, fontWeight: 600, letterSpacing: '.08em', textTransform: 'uppercase',
  color: 'var(--text-hint)', marginBottom: 10, fontFamily: 'var(--font-body)',
}

const smallBtn = {
  display: 'inline-flex', alignItems: 'center', gap: 5, padding: '4px 10px', borderRadius: 20,
  border: '1px solid var(--border)', background: 'none', color: 'var(--text-muted)',
  fontSize: 11, cursor: 'pointer', fontFamily: 'var(--font-body)', flexShrink: 0,
}

function Empty({ children }) {
  return (
    <div style={{
      textAlign: 'center', padding: '32px 16px', background: 'var(--surface-2)',
      borderRadius: 'var(--radius-md)', border: '1px dashed var(--border)',
    }}>
      <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300 }}>{children}</p>
    </div>
  )
}

function Chip({ st, pl }) {
  return (
    <span style={{
      fontSize: 10, padding: '2px 10px', borderRadius: 20, flexShrink: 0,
      color: st.c, background: st.bg, border: `1px solid ${st.bd}`,
    }}>{st.l[pl ? 0 : 1]}</span>
  )
}

export function UserPanel({ user, onClose, credits = 0, onBuy }) {
  const { i18n } = useTranslation()
  const pl = i18n.language?.startsWith('pl')
  const locale = pl ? 'pl-PL' : 'en-GB'

  const [tab, setTab] = useState('analyses')
  const [analyses, setAnalyses] = useState(null)
  const [ledger, setLedger] = useState(null)
  const [gens, setGens] = useState(null)
  const [tickets, setTickets] = useState(null)
  const [selected, setSelected] = useState(null)
  const [deleting, setDeleting] = useState(null)
  const [ticketFor, setTicketFor] = useState(null)
  const [exporting, setExporting] = useState(false)
  const [notice, setNotice] = useState(null)
  const [confirmEmail, setConfirmEmail] = useState('')
  const [delState, setDelState] = useState('idle')
  const [delError, setDelError] = useState(null)
  const [pwOpen, setPwOpen] = useState(false)

  const date = iso => new Date(iso).toLocaleDateString(locale, { day: 'numeric', month: 'short', year: 'numeric' })
  const time = iso => new Date(iso).toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' })

  useEffect(() => { loadAnalyses() }, [])
  useEffect(() => {
    if (tab === 'credits' && ledger === null) loadLedger()
    if (tab === 'previews' && gens === null) loadGens()
    if ((tab === 'previews' || tab === 'account') && tickets === null) loadTickets()
  }, [tab])

  async function loadAnalyses() {
    try {
      const { data } = await supabase.auth.getSession()
      const token = data.session?.access_token
      if (!token) { setAnalyses([]); return }
      const res = await fetch(`${API_BASE}/history`, { headers: { Authorization: `Bearer ${token}` } })
      const d = res.ok ? await res.json() : { analyses: [] }
      setAnalyses(d.analyses || [])
    } catch (e) {
      console.error('History fetch error:', e)
      setAnalyses([])
    }
  }

  async function loadLedger() {
    const { data, error } = await supabase.from('credit_ledger')
      .select('id, delta, reason, note, created_at').order('created_at', { ascending: false }).limit(50)
    if (error) console.error('Ledger fetch error:', error)
    setLedger(data || [])
  }

  async function loadGens() {
    const { data, error } = await supabase.from('generations')
      .select('id, style_name, color_id, gender, status, created_at').order('created_at', { ascending: false }).limit(30)
    if (error) console.error('Generations fetch error:', error)
    setGens(data || [])
  }

  async function loadTickets() {
    try { setTickets(await fetchTickets()) }
    catch (e) { console.error('Tickets fetch error:', e); setTickets([]) }
  }

  async function handleDelete(id) {
    if (!window.confirm(pl ? 'Usunąć tę analizę?' : 'Delete this analysis?')) return
    setDeleting(id)
    try {
      const { data } = await supabase.auth.getSession()
      const token = data.session?.access_token
      await fetch(`${API_BASE}/history/${id}`, { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } })
      setAnalyses(prev => prev.filter(a => a.id !== id))
      if (selected?.id === id) setSelected(null)
    } catch (e) { console.error('Delete error:', e) }
    finally { setDeleting(null) }
  }

  async function handleExport() {
    setExporting(true); setNotice(null)
    try {
      await exportAccountData()
      setNotice({ ok: true, text: pl ? 'Plik z Twoimi danymi został pobrany.' : 'Your data file has been downloaded.' })
    } catch (e) {
      const code = parseError(e).code
      setNotice({ ok: false, text: code === 'RATE_LIMITED'
        ? (pl ? 'Eksport jest dostępny kilka razy na godzinę. Spróbuj później.' : 'Export is limited to a few times per hour. Try again later.')
        : (pl ? 'Nie udało się wyeksportować danych. Spróbuj ponownie później.' : 'Could not export your data. Please try again later.') })
    } finally { setExporting(false) }
  }

  async function handleSignOutOthers() {
    if (!window.confirm(pl
      ? 'Wylogować wszystkie inne urządzenia? To urządzenie pozostanie zalogowane.'
      : 'Sign out all other devices? This device will stay signed in.')) return
    const { error } = await supabase.auth.signOut({ scope: 'others' })
    setNotice(error
      ? { ok: false, text: pl ? 'Nie udało się wylogować innych urządzeń.' : 'Could not sign out other devices.' }
      : { ok: true,  text: pl ? 'Inne urządzenia zostały wylogowane.' : 'Other devices have been signed out.' })
  }

  async function handleDeleteAccount() {
    setDelState('working'); setDelError(null)
    try {
      await deleteAccount(confirmEmail.trim())
      setDelState('done')
      setTimeout(async () => {
        await supabase.auth.signOut({ scope: 'local' }).catch(() => {})   // lokalnie: konto już nie istnieje na serwerze
        window.location.assign('/')
      }, 2500)
    } catch (e) {
      const code = parseError(e).code
      setDelState('idle')
      setDelError(
        code === 'EMAIL_MISMATCH' ? (pl ? 'Podany e-mail nie jest powiązany z żadnym kontem.' : 'The email does not match your account.')
        : code === 'RATE_LIMITED' ? (pl ? 'Za dużo prób. Spróbuj później.' : 'Too many attempts. Try again later.')
        : (pl ? 'Nie udało się usunąć konta. Spróbuj ponownie.' : 'Could not delete the account. Please try again.'))
    }
  }

  const ticketByGen = new Map(
    (tickets || []).filter(t => t.generation_id).map(t => [t.generation_id, t]))
  const emailMatches = confirmEmail.trim().toLowerCase() === (user.email || '').toLowerCase()

  const tabs = [
    { id: 'analyses', Icon: History, label: pl ? 'Analizy' : 'Analyses' },
    { id: 'credits', Icon: Coins, label: pl ? 'Kredyty' : 'Credits' },
    { id: 'previews', Icon: WandSparkles, label: pl ? 'Podglądy' : 'Previews' },
    { id: 'account', Icon: User, label: pl ? 'Konto' : 'Account' },
  ]

  return (
    <>
      <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.55)', zIndex: 200 }} />
      <div role="dialog" aria-modal="true" style={{
        position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 201,
        width: 'calc(100% - 32px)', maxWidth: 760, height: 'min(640px, 86vh)',
        background: 'var(--surface)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)',
        boxShadow: '0 24px 64px rgba(0,0,0,.25)', overflow: 'hidden',
        display: 'flex', flexDirection: 'column', animation: 'modalIn .2s ease',
      }}>
        {/* header */}
        <div style={{ padding: '18px 24px 0', borderBottom: '1px solid var(--border)', position: 'relative', flexShrink: 0 }}>
          <button onClick={onClose} aria-label="close" style={{
            position: 'absolute', top: 16, right: 16, background: 'none', border: 'none',
            cursor: 'pointer', color: 'var(--text-hint)', display: 'flex',
          }}><X size={18} strokeWidth={1.5} /></button>

          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 500, color: 'var(--text)', marginBottom: 2 }}>
            {pl ? 'Twoje konto' : 'Your account'}
          </h2>
          <p style={{ fontSize: 11, color: 'var(--text-hint)', fontFamily: 'var(--font-mono)' }}>{user.email}</p>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, margin: '14px 0' }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 6, padding: '5px 12px', borderRadius: 20,
              background: 'var(--accent-soft)', border: '1px solid var(--accent)', color: 'var(--accent)',
              fontSize: 12, fontWeight: 500,
            }}>
              <Coins size={13} strokeWidth={1.5} />
              {credits} {creditWord(credits, pl)}
            </span>
            <button onClick={onBuy} style={{ ...btnAccent, fontSize: 11, padding: '5px 12px', minWidth: 'auto' }}>
              {pl ? 'Kup kredyty' : 'Buy credits'}
            </button>
          </div>

          <div role="tablist" style={{ display: 'flex', gap: 2, overflowX: 'auto' }}>
            {tabs.map(({ id, Icon, label }) => (
              <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)} style={{
                display: 'inline-flex', alignItems: 'center', gap: 6, padding: '8px 12px', cursor: 'pointer',
                background: 'none', border: 'none', fontSize: 12, fontFamily: 'var(--font-body)', whiteSpace: 'nowrap',
                color: tab === id ? 'var(--accent)' : 'var(--text-muted)', fontWeight: tab === id ? 500 : 300,
                borderBottom: `2px solid ${tab === id ? 'var(--accent)' : 'transparent'}`, marginBottom: -1,
              }}>
                <Icon size={13} strokeWidth={1.5} />{label}
              </button>
            ))}
          </div>
        </div>

        {/* body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: 16, minHeight: 0, textAlign: 'left' }}>

          {tab === 'analyses' && (
            analyses === null ? <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>{pl ? 'Ładowanie…' : 'Loading…'}</p>
            : analyses.length === 0 ? <Empty>{pl ? 'Nie masz jeszcze zapisanych analiz.' : 'No saved analyses yet.'}</Empty>
            : (
              <div className={`up-grid${selected ? ' has-detail' : ''}`}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0 }}>
                  <p style={sectionLabel}>{pl ? `Historia analiz (${analyses.length})` : `Analysis history (${analyses.length})`}</p>
                  {analyses.map(a => {
                    const active = selected?.id === a.id
                    return (
                      <div key={a.id} role="button" tabIndex={0}
                        onClick={() => setSelected(active ? null : a)}
                        onKeyDown={e => (e.key === 'Enter') && setSelected(active ? null : a)}
                        style={{
                          background: active ? 'var(--accent-soft)' : 'var(--surface-2)',
                          border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
                          borderRadius: 'var(--radius-sm)', padding: '10px 12px', cursor: 'pointer',
                          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8,
                        }}>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <span style={{ fontSize: 11, color: 'var(--text-hint)', display: 'block', marginBottom: 6 }}>
                            {date(a.created_at)} · {time(a.created_at)} · {a.gender === 'Woman' ? (pl ? 'Kobieta' : 'Woman') : (pl ? 'Mężczyzna' : 'Man')}
                          </span>
                          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                            {a.top_styles?.slice(0, 3).map(s => (
                              <span key={s.name} style={{
                                fontSize: 10, padding: '2px 8px', borderRadius: 20, background: 'var(--surface)',
                                border: '1px solid var(--border)', color: 'var(--text-muted)',
                              }}>{s.name}</span>
                            ))}
                          </div>
                        </div>
                        <button onClick={e => { e.stopPropagation(); handleDelete(a.id) }}
                          disabled={deleting === a.id} title={pl ? 'Usuń' : 'Delete'} aria-label={pl ? 'Usuń' : 'Delete'}
                          style={{
                            background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-hint)',
                            padding: 2, display: 'flex', opacity: deleting === a.id ? 0.4 : 1,
                          }}>
                          <Trash size={13} strokeWidth={1.5} />
                        </button>
                      </div>
                    )
                  })}
                </div>

                {selected && (
                  <div style={{ minWidth: 0 }}>
                    <p style={sectionLabel}>{pl ? 'Szczegóły analizy' : 'Analysis details'}</p>

                    {selected.top_styles?.length > 0 && (
                      <div style={{ marginBottom: 18 }}>
                        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                          {pl ? 'Rekomendowane fryzury' : 'Recommended hairstyles'}
                        </p>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                          {selected.top_styles.map(s => (
                            <span key={s.name} style={{
                              display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 11, padding: '5px 10px',
                              borderRadius: 20, background: 'var(--accent-soft)', border: '1px solid var(--accent)',
                              color: 'var(--accent)',
                            }}>
                              {s.name}
                              {s.score ? <b style={{ fontFamily: 'var(--font-mono)', fontSize: 10 }}>{s.score}/100</b> : null}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {TRAIT_GROUPS.map(g => {
                      const rows = g.keys.filter(k => selected.traits?.[k])
                      if (rows.length === 0) return null
                      return (
                        <div key={g.id} style={{ marginBottom: 16 }}>
                          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{pl ? g.pl : g.en}</p>
                          {rows.map(k => (
                            <div key={k} style={{
                              display: 'flex', justifyContent: 'space-between', gap: 12, padding: '6px 0',
                              borderBottom: '1px solid var(--border)', fontSize: 12,
                            }}>
                              <span style={{ color: 'var(--text-hint)' }}>{traitLabel(k, pl)}</span>
                              <span style={{ color: 'var(--text)', fontWeight: 500, textAlign: 'right' }}>
                                {traitValue(k, selected.traits[k], pl)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          )}

          {tab === 'credits' && (
            ledger === null ? <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>{pl ? 'Ładowanie…' : 'Loading…'}</p>
            : ledger.length === 0 ? <Empty>{pl ? 'Brak operacji na kredytach.' : 'No credit activity yet.'}</Empty>
            : (
              <div>
                <p style={sectionLabel}>{pl ? 'Historia kredytów' : 'Credit history'}</p>
                {ledger.map(r => (
                  <div key={r.id} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12,
                    padding: '10px 2px', borderBottom: '1px solid var(--border)',
                  }}>
                    <div style={{ minWidth: 0, flex: 1, textAlign: 'left' }}>
                      <p style={{ fontSize: 13, color: 'var(--text)', textAlign: 'left' }}>{ledgerLabel(r, pl)}</p>
                      <p style={{ fontSize: 11, color: 'var(--text-hint)', textAlign: 'left' }}>{date(r.created_at)} · {time(r.created_at)}</p>
                    </div>
                    <span style={{
                      fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 600, flexShrink: 0,
                      color: r.delta > 0 ? '#2d8f4e' : 'var(--text)',
                    }}>{r.delta > 0 ? `+${r.delta}` : r.delta}</span>
                  </div>
                ))}
              </div>
            )
          )}

          {tab === 'previews' && (
            gens === null ? <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>{pl ? 'Ładowanie…' : 'Loading…'}</p>
            : gens.length === 0 ? <Empty>{pl ? 'Nie wygenerowano jeszcze żadnego podglądu.' : 'No previews generated yet.'}</Empty>
            : (
              <div>
                <p style={sectionLabel}>{pl ? 'Historia podglądów' : 'Preview history'}</p>
                {gens.map(g => {
                  const st = GEN_STATUS[g.status] || GEN_STATUS.pending
                  const tk = ticketByGen.get(g.id)
                  const autoRefunded = g.status === 'failed' || g.status === 'refunded'
                  return (
                    <div key={g.id} style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12,
                      padding: '10px 2px', borderBottom: '1px solid var(--border)',
                    }}>
                      <div style={{ minWidth: 0, flex: 1, textAlign: 'left' }}>
                        <p style={{ fontSize: 13, color: 'var(--text)' }}>
                          {g.style_name} · {(COLORS[g.color_id] || [g.color_id, g.color_id])[pl ? 0 : 1]}
                        </p>
                        <p style={{ fontSize: 11, color: 'var(--text-hint)' }}>{date(g.created_at)} · {time(g.created_at)}</p>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
                        <Chip st={st} pl={pl} />
                        {tk ? (
                          <span style={{ fontSize: 10, color: 'var(--text-hint)' }}>
                            {pl ? 'Zgłoszono' : 'Reported'} · {(TICKET_STATUS[tk.status] || TICKET_STATUS.open).l[pl ? 0 : 1]}
                          </span>
                        ) : autoRefunded ? (
                          <span style={{ fontSize: 10, color: 'var(--text-hint)' }}>
                            {pl ? 'Kredyty zwrócone' : 'Credits refunded'}
                          </span>
                        ) : g.status !== 'pending' && (
                          <button onClick={() => setTicketFor({ id: g.id, label: g.style_name })} style={smallBtn}
                            title={pl ? 'Zgłoś problem z tym podglądem' : 'Report a problem with this preview'}>
                            {pl ? 'Zgłoś' : 'Report'}
                          </button>
                        )}
                      </div>
                    </div>
                  )
                })}
                <p style={{ fontSize: 11, color: 'var(--text-hint)', marginTop: 14, lineHeight: 1.6, fontWeight: 300 }}>
                  {pl ? 'Za nieudane podglądy kredyty zwracane są automatycznie.'
                      : 'Credits for failed previews are refunded automatically.'}
                </p>
              </div>
            )
          )}

          {tab === 'account' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
              {notice && (
                <p role="status" style={{
                  fontSize: 12, padding: '8px 12px', borderRadius: 'var(--radius-sm)',
                  color: notice.ok ? '#2d8f4e' : '#c0392b', background: notice.ok ? '#f0faf4' : '#fef4f2',
                  border: `1px solid ${notice.ok ? '#b7dfc7' : '#f5c6bc'}`,
                }}>{notice.text}</p>
              )}

              {/* reports */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <p style={{ ...sectionLabel, marginBottom: 0 }}>{pl ? 'Moje zgłoszenia' : 'My requests'}</p>
                  <button onClick={() => setTicketFor({ id: null, label: '' })} style={smallBtn}>
                    <LifeBuoy size={12} strokeWidth={1.5} />{pl ? 'Nowe zgłoszenie' : 'New request'}
                  </button>
                </div>
                {tickets === null ? <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>{pl ? 'Ładowanie…' : 'Loading…'}</p>
                : tickets.length === 0 ? <Empty>{pl ? 'Nie masz żadnych zgłoszeń.' : 'You have no requests.'}</Empty>
                : tickets.map(t => (
                    <div key={t.id} style={{ padding: '10px 2px', borderBottom: '1px solid var(--border)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
                        <p style={{ fontSize: 13, color: 'var(--text)' }}>
                          {(CATEGORY_LABEL[t.category] || [t.category, t.category])[pl ? 0 : 1]}
                        </p>
                        <Chip st={TICKET_STATUS[t.status] || TICKET_STATUS.open} pl={pl} />
                      </div>
                      <p style={{ fontSize: 11, color: 'var(--text-hint)', margin: '2px 0 4px' }}>
                        {date(t.created_at)} · {time(t.created_at)}
                      </p>
                      <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.5 }}>
                        {t.message.length > 140 ? `${t.message.slice(0, 140)}…` : t.message}
                      </p>
                      {t.credits_granted > 0 && (
                        <p style={{ fontSize: 11, color: '#2d8f4e', marginTop: 4 }}>
                          {pl ? `Przyznano ${t.credits_granted} ${creditWord(t.credits_granted, true)}`
                              : `${t.credits_granted} ${creditWord(t.credits_granted, false)} granted`}
                        </p>
                      )}
                    </div>
                  ))}
              </div>

              {/* data export */}
              <div>
                <p style={sectionLabel}>{pl ? 'Twoje dane' : 'Your data'}</p>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.6, marginBottom: 10 }}>
                  {pl ? 'Pobierz plik JSON z danymi: historią analiz, podglądów, kredytów i zgłoszeń.'
                      : 'Download a JSON file with your data: analysis, preview, credit and request history.'}
                </p>
                <button onClick={handleExport} disabled={exporting} style={smallBtn}>
                  <Download size={12} strokeWidth={1.5} />
                  {exporting ? (pl ? 'Pobieranie…' : 'Downloading…') : (pl ? 'Pobierz moje dane' : 'Download my data')}
                </button>
              </div>

              {/* security */}
              <div>
                <p style={sectionLabel}>{pl ? 'Bezpieczeństwo' : 'Security'}</p>
                <p style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 300, lineHeight: 1.6, marginBottom: 10 }}>
                  {pl ? 'Jeśli konto powstało przez link z e-maila, nie ma jeszcze hasła. Ustaw je tutaj, aby logować się hasłem.'
                      : 'If your account was created via an email link, it has no password yet. Set one here to sign in with a password.'}
                </p>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button onClick={() => setPwOpen(true)} style={smallBtn}>
                    <KeyRound size={12} strokeWidth={1.5} />{pl ? 'Ustaw lub zmień hasło' : 'Set or change password'}
                  </button>
                  <button onClick={handleSignOutOthers} style={smallBtn}>
                    <LogOut size={12} strokeWidth={1.5} />{pl ? 'Wyloguj inne urządzenia' : 'Sign out other devices'}
                  </button>
                </div>
              </div>

              {/* account deleting */}
              {/* TODO prawnik */}
              <div style={{ border: '1px solid #f5c6bc', background: '#fef4f2', borderRadius: 'var(--radius-md)', padding: 16 }}>
                {delState === 'done' ? (
                  <p style={{ fontSize: 13, color: '#c0392b' }}>
                    {pl ? 'Konto zostało usunięte. Za chwilę przekierujemy Cię na stronę główną.'
                        : 'Your account has been deleted. Redirecting to the home page…'}
                  </p>
                ) : (
                  <>
                    <p style={{ ...sectionLabel, color: '#c0392b' }}>{pl ? 'Usuń konto' : 'Delete account'}</p>
                    <p style={{ fontSize: 12, color: '#7a2a20', lineHeight: 1.6, marginBottom: 10 }}>
                      {pl
                        ? `Usunięcie konta jest nieodwracalne. Usuniemy historię analiz, oceny, zgłoszenia i dane logowania. Niewykorzystane kredyty (masz ${credits}) przepadają bez zwrotu.`
                        : `Deleting your account is permanent. We will erase your analysis history, ratings, requests and login data. Unused credits (you have ${credits}) are forfeited without refund.`}
                    </p>
                    <label style={{ fontSize: 11, color: '#7a2a20', display: 'block', marginBottom: 4 }}>
                      {pl ? `Aby potwierdzić, wpisz swój e-mail: ${user.email}` : `To confirm, type your email: ${user.email}`}
                    </label>
                    <input type="email" value={confirmEmail} onChange={e => setConfirmEmail(e.target.value)}
                      autoComplete="off" style={{
                        width: '100%', padding: '9px 10px', marginBottom: 10, border: '1px solid #f5c6bc',
                        borderRadius: 'var(--radius-md)', background: '#fff', color: '#222', fontSize: 13, outline: 'none',
                      }} />
                    {delError && <p style={{ fontSize: 12, color: '#c0392b', marginBottom: 8 }}>{delError}</p>}
                    <button onClick={handleDeleteAccount} disabled={!emailMatches || delState === 'working'} style={{
                      padding: '9px 16px', borderRadius: 'var(--radius-md)', border: 'none', fontSize: 12, fontWeight: 500,
                      background: '#c0392b', color: '#fff', fontFamily: 'var(--font-body)',
                      cursor: emailMatches && delState !== 'working' ? 'pointer' : 'default',
                      opacity: emailMatches && delState !== 'working' ? 1 : 0.5,
                    }}>
                      {delState === 'working' ? (pl ? 'Usuwanie…' : 'Deleting…') : (pl ? 'Usuń konto na stałe' : 'Delete account permanently')}
                    </button>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {ticketFor && (
        <TicketModal
          generationId={ticketFor.id}
          styleLabel={ticketFor.label}
          onClose={() => setTicketFor(null)}
          onCreated={loadTickets}
        />
      )}

      {pwOpen && <SetPasswordModal mode="change" onClose={() => setPwOpen(false)} />}

      <style>{`
        .up-grid { display: grid; grid-template-columns: 1fr; gap: 20px; }
        .up-grid.has-detail { grid-template-columns: 300px 1fr; }
        @media (max-width: 640px) { .up-grid.has-detail { grid-template-columns: 1fr; } }
      `}</style>
    </>
  )
}