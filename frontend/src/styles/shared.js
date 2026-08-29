export const btnAccent = {
  background:    'var(--accent)',
  color:         '#fff',
  border:        'none',
  borderRadius:  'var(--radius-md)',
  padding:       '9px 18px',
  fontSize:      13,
  fontFamily:    'var(--font-body)',
  fontWeight:    500,
  cursor:        'pointer',
  letterSpacing: '.02em',
  transition:    'background .15s',
  height:        34,
  display:       'inline-flex',
  alignItems:    'center',
  whiteSpace:    'nowrap',
}

export const btnOutline = {
  background:    'none',
  color:         'var(--text-muted)',
  border:        '1px solid var(--border)',
  borderRadius:  'var(--radius-sm)',
  padding:       '0 12px',
  fontSize:      12,
  fontFamily:    'var(--font-body)',
  cursor:        'pointer',
  letterSpacing: '.02em',
  transition:    'border-color .15s, color .15s',
  height:        34,
  display:       'inline-flex',
  alignItems:    'center',
  whiteSpace:    'nowrap',
}

export const sectionH2 = {
  fontFamily: 'var(--font-display)',
  fontSize: 'clamp(22px, 3vw, 30px)',
  fontWeight: 500,
  letterSpacing: '.01em',
  color: 'var(--text)',
  marginBottom: 8,
}

export const darkToggleBtn = (dark) => ({
  ...btnOutline,
  gap: 6,
  padding: '0 10px',
})

export const darkToggleTrack = (dark) => ({
  width:      28, height: 14,
  borderRadius: 7,
  background: dark ? 'var(--accent)' : 'var(--border)',
  position:   'relative', display: 'block',
  transition: 'background .2s', flexShrink: 0,
})

export const darkToggleKnob = (dark) => ({
  position:     'absolute',
  top:          2,
  left:         dark ? 14 : 2,
  width:        10, height: 10,
  borderRadius: '50%',
  background:   '#fff',
  transition:   'left .2s',
})