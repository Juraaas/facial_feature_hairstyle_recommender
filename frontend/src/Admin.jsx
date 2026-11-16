import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from './hooks/useAuth'
import { useDarkMode } from './hooks/useDarkMode'
import { adminApi, parseError } from './api/client'

const fmt = iso => iso
  ? new Date(iso).toLocaleString('pl-PL', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })
  : '-'

const LEDGER = {
  signup_bonus: 'Bonus na start', purchase: 'Zakup pakietu', spend_analysis: 'Pełna analiza',
  spend_preview: 'Podgląd fryzury', refund_failed: 'Zwrot za nieudaną operację',
  refund_ticket: 'Zwrot po zgłoszeniu', admin_grant: 'Kredyty od admina', admin_revoke: 'Korekta salda',
}
const CATEGORY = {
  preview_quality: 'Jakość podglądu', technical: 'Problem techniczny', billing: 'Płatności i kredyty', other: 'Inne',
}
const GEN_STATUS = { succeeded: ['Udany', 'ok'], failed: ['Nieudany', 'bad'], pending: ['W toku', 'warn'], refunded: ['Zwrócony', 'muted'] }
const TICKET_STATUS = {
  open: ['Otwarte', 'warn'], in_progress: ['W trakcie', 'warn'], resolved: ['Rozpatrzone', 'ok'], rejected: ['Odrzucone', 'muted'],
}
const TICKET_ERRORS = {
  ALREADY_REFUNDED: 'Za ten podgląd kredyty zostały już zwrócone (automatycznie lub wcześniej).',
  AMOUNT_EXCEEDS_CHARGE: 'Zwrot nie może przekroczyć kwoty pobranej za ten podgląd.',
  TICKET_CLOSED: 'To zgłoszenie jest już zamknięte. Odśwież listę.',
  TICKET_NO_OWNER: 'Zgłoszenie nie ma właściciela (konto zostało usunięte).',
  INVALID_AMOUNT: 'Nieprawidłowa liczba kredytów (od 1 do 20).',
  TICKET_NOT_FOUND: 'Nie znaleziono zgłoszenia.',
  FORBIDDEN: 'Brak uprawnień administratora.',
}

const S = {
  card: { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: 16 },
  label: { fontSize: 10, fontWeight: 600, letterSpacing: '.08em', textTransform: 'uppercase',
    color: 'var(--text-hint)', marginBottom: 8, fontFamily: 'var(--font-body)' },
  input: { padding: '8px 10px', border: '1px solid var(--border)', borderRadius: 8, background: 'var(--surface-2)',
    color: 'var(--text)', fontSize: 13, outline: 'none', fontFamily: 'var(--font-body)' },
  muted: { fontSize: 12, color: 'var(--text-muted)', fontWeight: 300 },
}

const TONES = {
  ok:    { c: '#2d8f4e', bg: '#f0faf4', bd: '#b7dfc7' },
  warn:  { c: '#C8975A', bg: '#fef8f0', bd: '#f0d5a8' },
  bad:   { c: '#c0392b', bg: '#fef4f2', bd: '#f5c6bc' },
  muted: { c: 'var(--text-muted)', bg: 'var(--surface-2)', bd: 'var(--border)' },
}

function Chip({ tone = 'muted', children }) {
  const t = TONES[tone]
  return (
    <span style={{ fontSize: 10, padding: '2px 10px', borderRadius: 20, color: t.c, background: t.bg,
      border: `1px solid ${t.bd}`, whiteSpace: 'nowrap' }}>{children}</span>
  )
}

function Btn({ kind = 'ghost', style, ...props }) {
  const kinds = {
    primary: { background: 'var(--accent)', color: '#fff', border: 'none' },
    danger: { background: '#c0392b', color: '#fff', border: 'none' },
    ghost: { background: 'none', color: 'var(--text-muted)', border: '1px solid var(--border)' },
  }
  return (
    <button {...props} style={{
      padding: '7px 14px', borderRadius: 8, fontSize: 12, fontFamily: 'var(--font-body)',
      cursor: props.disabled ? 'default' : 'pointer', opacity: props.disabled ? 0.5 : 1,
      ...kinds[kind], ...style,
    }} />
  )
}

function ErrBox({ error, onRetry }) {
  return (
    <div style={{ ...S.card, borderColor: '#f5c6bc', background: '#fef4f2' }}>
      <p style={{ fontSize: 13, color: '#c0392b' }}>
        Błąd: {error.code}{error.message ? ` · ${error.message}` : ''}
      </p>
      {onRetry && <Btn onClick={onRetry} style={{ marginTop: 10 }}>Spróbuj ponownie</Btn>}
    </div>
  )
}

function Empty({ children }) {
  return (
    <div style={{ textAlign: 'center', padding: '32px 16px', background: 'var(--surface-2)',
      borderRadius: 12, border: '1px dashed var(--border)' }}>
      <p style={S.muted}>{children}</p>
    </div>
  )
}

function useLoad(fn, deps = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null })
  const run = useCallback(async () => {
    setState(s => ({ ...s, loading: true, error: null }))
    try {
      const data = await fn()
      setState({ data, loading: false, error: null })
    } catch (e) {
      setState({ data: null, loading: false, error: parseError(e) })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  useEffect(() => { run() }, [run])
  return { ...state, reload: run }
}

function Stat({ title, value, sub, tone }) {
  const color = tone === 'bad' ? '#c0392b' : tone === 'warn' ? '#C8975A' : 'var(--text)'
  return (
    <div style={S.card}>
      <p style={S.label}>{title}</p>
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 700, color, lineHeight: 1 }}>{value}</p>
      {sub && <p style={{ fontSize: 11, color: 'var(--text-hint)', marginTop: 8, lineHeight: 1.5 }}>{sub}</p>}
    </div>
  )
}

function Overview({ stats, onRefresh }) {
  if (!stats) return null
  const g = stats.generations
  const l = g.last_24h || {}
  const total24 = (l.succeeded || 0) + (l.failed || 0) + (l.pending || 0)
  const r = stats.ratings_7d
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
        <Btn onClick={onRefresh}>Odśwież</Btn>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 12 }}>
        <Stat title="Użytkownicy" value={stats.users.total} sub={`+${stats.users.new_7d} w ostatnich 7 dniach`} />
        <Stat title="Analizy" value={stats.analyses.total} sub={`${stats.analyses.last_24h} w ostatnich 24 h`} />
        <Stat title="Podglądy (24 h)" value={total24}
          sub={`udane ${l.succeeded || 0} · nieudane ${l.failed || 0} · w toku ${l.pending || 0}`}
          tone={(l.failed || 0) > 0 ? 'warn' : undefined} />
        <Stat title="Zawieszone podglądy" value={g.stuck} sub="w toku dłużej niż 10 minut" tone={g.stuck > 0 ? 'bad' : undefined} />
        <Stat title="Otwarte zgłoszenia" value={stats.tickets.open} tone={stats.tickets.open > 0 ? 'warn' : undefined} />
        <Stat title="Oceny podglądów (7 dni)" value={`${r.good} / ${r.bad}`} sub="dobre / słabe" />
        <Stat title="Sprzedaż (30 dni)" value={`${Number(stats.sales_30d.revenue_pln).toFixed(2)} zł`}
          sub={`${stats.sales_30d.purchases} zakupów · ${stats.sales_30d.credits} kredytów (tylko płatności w PLN)`} />
      </div>
    </div>
  )
}

function TicketCard({ t, notify, onDone }) {
  const isOpen = t.status === 'open' || t.status === 'in_progress'
  const g = t.generation
  const refundedAlready = (t.refunded || 0) > 0 || (g && (g.status === 'failed' || g.status === 'refunded'))
  const [credits, setCredits] = useState(String(Math.min(t.charged ?? 3, 20)))
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)

  async function act(action) {
    const n = parseInt(credits, 10)
    if (action === 'grant') {
      if (!(n >= 1 && n <= 20)) { notify({ ok: false, text: 'Podaj liczbę kredytów od 1 do 20.' }); return }
      if (!window.confirm(`Przyznać ${n} kredytów użytkownikowi ${t.user_email}?`)) return
    } else if (!window.confirm('Odrzucić zgłoszenie bez zwrotu?')) return

    setBusy(true)
    try {
      await adminApi.resolveTicket(t.id, { action, credits: action === 'grant' ? n : 0, note: note.trim() })
      notify({ ok: true, text: action === 'grant' ? `Przyznano ${n} kredytów, zgłoszenie rozpatrzone.` : 'Zgłoszenie odrzucone.' })
      onDone()
    } catch (e) {
      const c = parseError(e)
      notify({ ok: false, text: TICKET_ERRORS[c.code] || `Błąd: ${c.code}` })
    } finally { setBusy(false) }
  }

  const [stLabel, stTone] = TICKET_STATUS[t.status] || TICKET_STATUS.open
  return (
    <div style={{ ...S.card, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <p style={{ fontSize: 14, fontWeight: 500 }}>{CATEGORY[t.category] || t.category}</p>
          <p style={{ fontSize: 11, color: 'var(--text-hint)' }}>{t.user_email || 'konto usunięte'} · {fmt(t.created_at)}</p>
        </div>
        <Chip tone={stTone}>{stLabel}</Chip>
      </div>

      <p style={{ fontSize: 13, lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{t.message}</p>

      {g && (
        <div style={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px' }}>
          <p style={S.label}>Podgląd, którego dotyczy zgłoszenie</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', fontSize: 12 }}>
            <span>{g.style_name} · {g.color_id}</span>
            <Chip tone={(GEN_STATUS[g.status] || GEN_STATUS.pending)[1]}>{(GEN_STATUS[g.status] || GEN_STATUS.pending)[0]}</Chip>
            {g.error_code && <Chip tone="bad">{g.error_code}</Chip>}
            {g.duration_ms != null && <span style={S.muted}>{(g.duration_ms / 1000).toFixed(1)} s</span>}
            {t.rating && <Chip tone={t.rating === 'good' ? 'ok' : 'bad'}>ocena użytkownika: {t.rating === 'good' ? 'dobrze' : 'słabo'}</Chip>}
            <span style={S.muted}>pobrano: {t.charged ?? '–'} · zwrócono: {t.refunded || 0}</span>
          </div>
          {refundedAlready && isOpen && (
            <p style={{ fontSize: 11, color: '#c0392b', marginTop: 8 }}>
              Kredyty za ten podgląd zostały już zwrócone: przyznanie kolejnych zostanie zablokowane (możesz tylko odrzucić zgłoszenie).
            </p>
          )}
        </div>
      )}

      {isOpen ? (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <input type="number" min="1" max={t.charged ?? 20} value={credits} onChange={e => setCredits(e.target.value)}
            style={{ ...S.input, width: 80 }} aria-label="Liczba kredytów" />
          <input value={note} onChange={e => setNote(e.target.value)} maxLength={500} placeholder="Notatka (opcjonalnie)"
            style={{ ...S.input, flex: 1, minWidth: 160 }} />
          <Btn kind="primary" disabled={busy || refundedAlready} onClick={() => act('grant')}>Przyznaj kredyty</Btn>
          <Btn disabled={busy} onClick={() => act('reject')}>Odrzuć</Btn>
        </div>
      ) : (
        <p style={S.muted}>
          {t.resolution === 'credits_granted' ? `Przyznano ${t.credits_granted} kredytów` : 'Odrzucone bez zwrotu'}
          {t.resolved_at ? ` · ${fmt(t.resolved_at)}` : ''}{t.admin_note ? ` · notatka: ${t.admin_note}` : ''}
        </p>
      )}
    </div>
  )
}

function Tickets({ notify, onChanged }) {
  const [status, setStatus] = useState('open')
  const { data, loading, error, reload } = useLoad(() => adminApi.tickets(status), [status])
  const tickets = data?.tickets ?? []
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', gap: 6 }}>
        {[['open', 'Otwarte'], ['closed', 'Zamknięte'], ['all', 'Wszystkie']].map(([id, label]) => (
          <Btn key={id} kind={status === id ? 'primary' : 'ghost'} onClick={() => setStatus(id)}>{label}</Btn>
        ))}
      </div>
      {error && <ErrBox error={error} onRetry={reload} />}
      {loading && !data && <p style={S.muted}>Ładowanie…</p>}
      {data && tickets.length === 0 && <Empty>Brak zgłoszeń w tej kategorii.</Empty>}
      {tickets.map(t => (
        <TicketCard key={t.id} t={t} notify={notify} onDone={() => { reload(); onChanged?.() }} />
      ))}
    </div>
  )
}

function UserDetail({ id, notify, onClose, onChanged }) {
  const { data, loading, error, reload } = useLoad(() => adminApi.user(id), [id])
  const [amount, setAmount] = useState('')
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)

  async function adjust(sign) {
    const n = parseInt(amount, 10)
    if (!(n >= 1 && n <= 500)) { notify({ ok: false, text: 'Podaj liczbę kredytów od 1 do 500.' }); return }
    if (note.trim().length < 3) { notify({ ok: false, text: 'Podaj powód (min. 3 znaki). Trafi do księgi kredytów.' }); return }
    if (!window.confirm(`${sign > 0 ? 'Dodać' : 'Odjąć'} ${n} kredytów: ${data.user.email}?`)) return
    setBusy(true)
    try {
      const r = await adminApi.adjustCredits(id, sign * n, note.trim())
      notify({ ok: true, text: `Gotowe. Nowe saldo: ${r.balance}.` })
      setAmount(''); setNote(''); reload(); onChanged?.()
    } catch (e) {
      const c = parseError(e)
      notify({ ok: false, text: c.code === 'INSUFFICIENT_CREDITS'
        ? 'Użytkownik ma mniej kredytów, niż próbujesz odjąć.' : `Błąd: ${c.code}` })
    } finally { setBusy(false) }
  }

  if (error) return <ErrBox error={error} onRetry={reload} />
  if (loading && !data) return <p style={S.muted}>Ładowanie…</p>
  if (!data) return null
  const u = data.user
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div style={S.card}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
          <div>
            <p style={{ fontSize: 15, fontWeight: 500, wordBreak: 'break-all' }}>{u.email}</p>
            <p style={{ fontSize: 11, color: 'var(--text-hint)' }}>od {fmt(u.created_at)} · {data.analyses_count} analiz</p>
          </div>
          <Btn onClick={onClose}>Zamknij</Btn>
        </div>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: 26, fontWeight: 700, color: 'var(--accent)', margin: '12px 0 4px' }}>
          {u.credits} <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--text-muted)' }}>kredytów</span>
        </p>
        <p style={S.label}>Zmiana salda</p>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <input type="number" min="1" max="500" value={amount} onChange={e => setAmount(e.target.value)}
            placeholder="Ilość" style={{ ...S.input, width: 90 }} />
          <input value={note} onChange={e => setNote(e.target.value)} maxLength={200}
            placeholder="Powód (wymagany)" style={{ ...S.input, flex: 1, minWidth: 160 }} />
          <Btn kind="primary" disabled={busy} onClick={() => adjust(1)}>Dodaj</Btn>
          <Btn kind="danger" disabled={busy} onClick={() => adjust(-1)}>Odejmij</Btn>
        </div>
      </div>

      <div style={S.card}>
        <p style={S.label}>Księga kredytów</p>
        {data.ledger.length === 0 ? <p style={S.muted}>Brak wpisów.</p> : data.ledger.map(r => (
          <div key={r.id} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
            <div>
              <span>{LEDGER[r.reason] || r.reason}</span>
              {r.note && <span style={{ color: 'var(--text-hint)' }}> · {r.note}</span>}
              <p style={{ fontSize: 10, color: 'var(--text-hint)' }}>{fmt(r.created_at)}</p>
            </div>
            <b style={{ fontFamily: 'var(--font-mono)', color: r.delta > 0 ? '#2d8f4e' : 'var(--text)' }}>{r.delta > 0 ? `+${r.delta}` : r.delta}</b>
          </div>
        ))}
      </div>

      <div style={S.card}>
        <p style={S.label}>Ostatnie podglądy</p>
        {data.generations.length === 0 ? <p style={S.muted}>Brak podglądów.</p> : data.generations.map(g => (
          <div key={g.id} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
            <div>
              <span>{g.style_name} · {g.color_id}</span>
              <p style={{ fontSize: 10, color: 'var(--text-hint)' }}>{fmt(g.created_at)}{g.error_code ? ` · ${g.error_code}` : ''}</p>
            </div>
            <Chip tone={(GEN_STATUS[g.status] || GEN_STATUS.pending)[1]}>{(GEN_STATUS[g.status] || GEN_STATUS.pending)[0]}</Chip>
          </div>
        ))}
      </div>

      <div style={S.card}>
        <p style={S.label}>Zgłoszenia</p>
        {data.tickets.length === 0 ? <p style={S.muted}>Brak zgłoszeń.</p> : data.tickets.map(t => (
          <div key={t.id} style={{ padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
              <span>{CATEGORY[t.category] || t.category}</span>
              <Chip tone={(TICKET_STATUS[t.status] || TICKET_STATUS.open)[1]}>{(TICKET_STATUS[t.status] || TICKET_STATUS.open)[0]}</Chip>
            </div>
            <p style={{ ...S.muted, marginTop: 2 }}>{t.message.length > 120 ? `${t.message.slice(0, 120)}…` : t.message}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

function Users({ notify }) {
  const [q, setQ] = useState('')
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState(null)
  const list = useLoad(() => adminApi.users(query), [query])
  const users = list.data?.users ?? []
  return (
    <div className="adm-two" style={{ display: 'grid', gridTemplateColumns: selected ? 'minmax(260px, 340px) 1fr' : '1fr', gap: 16, alignItems: 'start' }}>
      <div>
        <form onSubmit={e => { e.preventDefault(); setQuery(q.trim()) }} style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="Szukaj po e-mailu" style={{ ...S.input, flex: 1 }} />
          <Btn kind="primary" type="submit">Szukaj</Btn>
        </form>
        {list.error && <ErrBox error={list.error} onRetry={list.reload} />}
        {list.loading && !list.data && <p style={S.muted}>Ładowanie…</p>}
        {list.data && users.length === 0 && <Empty>Nie znaleziono użytkowników.</Empty>}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {users.map(u => (
            <button key={u.id} onClick={() => setSelected(u.id)} style={{
              ...S.card, padding: '10px 12px', textAlign: 'left', cursor: 'pointer', color: 'var(--text)',
              borderColor: selected === u.id ? 'var(--accent)' : 'var(--border)',
              background: selected === u.id ? 'var(--accent-soft)' : 'var(--surface)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                <span style={{ fontSize: 13, wordBreak: 'break-all' }}>{u.email}</span>
                <b style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--accent)' }}>{u.credits}</b>
              </div>
              <p style={{ fontSize: 10, color: 'var(--text-hint)' }}>{fmt(u.created_at)}{u.is_admin ? ' · admin' : ''}</p>
            </button>
          ))}
        </div>
      </div>
      {selected && <UserDetail key={selected} id={selected} notify={notify} onClose={() => setSelected(null)} onChanged={list.reload} />}
    </div>
  )
}

function Stuck({ notify, onChanged }) {
  const { data, loading, error, reload } = useLoad(() => adminApi.stuck(), [])
  const [busy, setBusy] = useState(null)
  const items = data?.generations ?? []

  async function fail(g) {
    if (!window.confirm(`Oznaczyć podgląd "${g.style_name}" (${g.user_email}) jako nieudany i zwrócić kredyty, jeśli zostały pobrane?`)) return
    setBusy(g.id)
    try {
      const r = await adminApi.failStuck(g.id)
      notify({ ok: true, text: r.refunded
        ? 'Podgląd oznaczony jako nieudany, kredyty zwrócone.'
        : 'Podgląd oznaczony jako nieudany (nie było pobrania kredytów).' })
      reload(); onChanged?.()
    } catch (e) {
      const c = parseError(e)
      notify({ ok: false, text: c.code === 'NOT_STUCK' ? 'Ten podgląd nie jest już zawieszony.' : `Błąd: ${c.code}` })
    } finally { setBusy(null) }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <p style={S.muted}>Podglądy w statusie „w toku" dłużej niż 10 minut, najczęściej po awarii serwera.</p>
      {error && <ErrBox error={error} onRetry={reload} />}
      {loading && !data && <p style={S.muted}>Ładowanie…</p>}
      {data && items.length === 0 && <Empty>Brak zawieszonych podglądów.</Empty>}
      {items.map(g => (
        <div key={g.id} style={{ ...S.card, display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <div>
            <p style={{ fontSize: 13 }}>{g.style_name} · {g.color_id}</p>
            <p style={{ fontSize: 11, color: 'var(--text-hint)' }}>{g.user_email || '–'} · {fmt(g.created_at)}</p>
          </div>
          <Btn kind="danger" disabled={busy === g.id} onClick={() => fail(g)}>Oznacz jako nieudany i zwróć</Btn>
        </div>
      ))}
    </div>
  )
}

export function Admin() {
  const { user } = useAuth()
  const [dark] = useDarkMode()
  const [tab, setTab] = useState('overview')
  const [waited, setWaited] = useState(false)
  const [toast, setToast] = useState(null)
  const uid = user?.id ?? null
  const stats = useLoad(() => (uid ? adminApi.stats() : Promise.resolve(null)), [uid])

  useEffect(() => { const t = setTimeout(() => setWaited(true), 1500); return () => clearTimeout(t) }, [])
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 6000)
    return () => clearTimeout(t)
  }, [toast])

  let access = 'ok'
  if (!uid) access = waited ? 'login' : 'checking'
  else if (stats.loading && !stats.data) access = 'checking'
  else if (stats.error) {
    access = stats.error.code === 'FORBIDDEN' ? 'denied'
           : stats.error.code === 'UNAUTHORIZED' ? 'login' : 'error'
  }

  const tabs = [
    { id: 'overview', label: 'Przegląd' },
    { id: 'tickets', label: 'Zgłoszenia', badge: stats.data?.tickets?.open },
    { id: 'users', label: 'Użytkownicy' },
    { id: 'stuck', label: 'Zawieszone podglądy', badge: stats.data?.generations?.stuck, bad: true },
  ]

  const message = {
    checking: 'Sprawdzanie dostępu…',
    login: 'Zaloguj się na konto administratora w aplikacji, a potem wróć na tę stronę.',
    denied: 'Brak dostępu. To konto nie ma uprawnień administratora.',
    error: 'Nie udało się połączyć z serwerem.',
  }[access]

  return (
    <div data-theme={dark ? 'dark' : 'light'} style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)', textAlign: 'left'}}>
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 24px',
        borderBottom: '1px solid var(--border)', background: 'var(--surface)', position: 'sticky', top: 0, zIndex: 10 }}>
        <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 17, fontWeight: 500, color: 'var(--text)', margin: 0 }}>Stylizzer Panel admina</h1>
        <Link to="/analyse" style={{ fontSize: 12, color: 'var(--accent)', textDecoration: 'none' }}>← Wróć do aplikacji</Link>
      </header>

      <main style={{ maxWidth: 1000, margin: '0 auto', padding: '20px 16px 64px' }}>
        {access !== 'ok' ? (
          <div style={{ ...S.card, textAlign: 'center', padding: '40px 20px' }}>
            <p style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: access === 'checking' ? 0 : 14 }}>{message}</p>
            {access === 'login' && <Link to="/analyse" style={{ color: 'var(--accent)', fontSize: 13 }}>Przejdź do aplikacji</Link>}
            {access === 'error' && <Btn onClick={stats.reload}>Spróbuj ponownie</Btn>}
          </div>
        ) : (
          <>
            {toast && (
              <div role="status" style={{
                display: 'flex', gap: 10, alignItems: 'center', padding: '10px 14px', marginBottom: 14, borderRadius: 10, fontSize: 13,
                color: toast.ok ? '#2d8f4e' : '#c0392b', background: toast.ok ? '#f0faf4' : '#fef4f2',
                border: `1px solid ${toast.ok ? '#b7dfc7' : '#f5c6bc'}`,
              }}>
                <span style={{ flex: 1 }}>{toast.text}</span>
                <button onClick={() => setToast(null)} aria-label="close" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'inherit' }}>✕</button>
              </div>
            )}

            <div role="tablist" style={{ display: 'flex', gap: 2, borderBottom: '1px solid var(--border)', marginBottom: 18, overflowX: 'auto' }}>
              {tabs.map(t => (
                <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)} style={{
                  padding: '9px 14px', background: 'none', border: 'none', cursor: 'pointer', whiteSpace: 'nowrap',
                  fontSize: 13, fontFamily: 'var(--font-body)', marginBottom: -1,
                  color: tab === t.id ? 'var(--accent)' : 'var(--text-muted)', fontWeight: tab === t.id ? 500 : 300,
                  borderBottom: `2px solid ${tab === t.id ? 'var(--accent)' : 'transparent'}`,
                }}>
                  {t.label}
                  {t.badge > 0 && (
                    <span style={{ marginLeft: 6, fontSize: 10, padding: '1px 7px', borderRadius: 20, color: '#fff',
                      background: t.bad ? '#c0392b' : '#C8975A' }}>{t.badge}</span>
                  )}
                </button>
              ))}
            </div>

            {tab === 'overview' && <Overview stats={stats.data} onRefresh={stats.reload} />}
            {tab === 'tickets' && <Tickets notify={setToast} onChanged={stats.reload} />}
            {tab === 'users' && <Users notify={setToast} />}
            {tab === 'stuck' && <Stuck notify={setToast} onChanged={stats.reload} />}
          </>
        )}
      </main>

      <style>{`@media (max-width: 760px) { .adm-two { grid-template-columns: 1fr !important; } }`}</style>
    </div>
  )
}