import { useState } from "react";
import { sendVote } from '../api/client'
import { useTranslation } from 'react-i18next'

const API_URL = import.meta.env.VITE_API_URL;

export function StyleCard({style, rank, features, gender, onReplace }) {
    const { t } = useTranslation()
    const [voted, setVoted] = useState(null)
    const score = style.display_score ?? Math.round(style.score * 100)
    const isTop = rank === 0
    const imgPath = style.image ? `${API_URL}/${style.image.replace(/^\/+/, "")}` : null

    async function handleVote(v) {
        setVoted(v)
        await sendVote(style.name, v, features, gender)
        if (v === 'down') onReplace(rank)
    }

    return (
        <div style={{
            background: 'var(--surface)',
            borderRadius: 'var(--radius-lg)',
            overflow: 'hidden',
            border: isTop ? '1.5px solid var(--accent)' : '1px solid var(--border)',
            display: 'flex',
            flexDirection: 'column',
            animation: 'fadeIn .3s ease',
            transition: 'box-shadow .2s',
        }}>
        {/* image */}
        <div style={{ position: 'relative', overflow: 'hidden' }}>
            {imgPath
            ? <img src={imgPath} alt={style.name} className="style-card-img" />
            : <div className="style-card-img" style={{ 
                background: 'var(--surface-2)', display: 'flex', 
                alignItems: 'center', justifyContent: 'center',
                fontSize: 32, color: 'var(--text-hint)'}}>✂</div>
            }
            {/* best match badge */}
            {isTop && (
            <div style={{
                position: 'absolute', top: 10, left: 10,
                background: 'var(--accent)', color: '#fff',fontSize: 9,
                fontWeight: 600, padding: '3px 9px', borderRadius: 20,
                letterSpacing: '.06em', textTransform: 'uppercase', fontFamily: 'var(--font-body)'
            }}>{t('best_match')}</div>
            )}
            {/* score badge */}
            <div style={{
            position: 'absolute', top: 10, right: 10, background: 'rgba(15,15,14,.72)', 
            backdropFilter: 'blur(6px)', color: '#fff', borderRadius: 20,
            padding: '3px 10px', fontSize: 11, fontFamily: 'var(--font-mono)',
            fontWeight: 400, letterSpacing: '.02em',
            }}>{score}%</div>
        </div>

        {/* body */}
        <div style={{ padding: '14px 14px 12px', display: 'flex',
            flexDirection: 'column', flex: 1, gap: 10 }}>
            {/* name + tags */}
            <div>
                <p style={{fontFamily: 'var(--font-display)', fontSize: 15,
                        fontWeight: 500, color: 'var(--text)', marginBottom: 6,
                        letterSpacing: '.01em'}}>{style.name}</p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {style.tags?.slice(0, 2).map(tag => (
                        <span key={tag} style={{
                        fontSize: 9, padding: '2px 8px', borderRadius: 20,
                        background: 'var(--surface-2)', color: 'var(--text-muted)',
                        border: '1px solid var(--border)', letterSpacing: '.04em',
                        textTransform: 'uppercase', fontFamily: 'var(--font-body)',
                        }}>{tag}</span>
                    ))}
                </div>
            </div>

            {/* description */}
            {style.description && (
            <p style={{fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.55,
                paddingBottom: 10, borderBottom: '1px solid var(--border)',
                display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical',
                overflow: 'hidden', fontWeight: 300}}>{style.description}</p>
            )}

            {/* why it works */}
            {style.contributions?.length > 0 && (
            <div>
                <p style={{fontSize: 9, fontWeight: 600, color: 'var(--text-hint)',
                textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 8, 
                fontFamily: 'var(--font-body)'}}>{t('why_it_works')}</p>
                {style.contributions.slice(0, 2).map((c, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                    <span style={{
                    fontSize: 11, color: 'var(--text-muted)', flexShrink: 0, width: 90, 
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 300 
                    }}>{c.desc}</span>
                    <div style={{ flex: 1, height: 2, borderRadius: 2, background: 'var(--border)'}}>
                    <div style={{
                        width: `${c.percent * 100}%`, height: '100%',
                        borderRadius: 2, background: 'var(--accent)'}}/>
                    </div>
                    <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)',
                        color:'var(--text-muted)', width: 28, textAlign: 'right',}}>
                    {Math.round(c.percent * 100)}%
                    </span>
                </div>
                ))}
            </div>
            )}

            {/* negatives */}
            {style.negatives?.length > 0 && (
            <div style={{
                fontSize: 11, padding: '6px 10px',
                borderRadius: 'var(--radius-sm)', background: 'var(--surface-2)',
                borderLeft: '2px solid var(--text-hint)', color: 'var(--text-muted)',
                fontWeight:  300, lineHeight:  1.5}}>
                ⚠ {style.negatives[0].reason} </div>
            )}

            {/* missing */}
            {style.missing?.length > 0 && (
            <div style={{
                fontSize: 11, padding: '6px 10px',
                borderRadius: 'var(--radius-sm)', background: 'var(--surface-2)',
                borderLeft: '2px solid var(--text-hint)', color: 'var(--text-muted)',
                fontWeight:  300, lineHeight:  1.5
            }}>
                ℹ {style.missing[0].reason}
            </div>
            )}

            {/* vote */}
            <div style={{ marginTop: 'auto', paddingTop: 4 }}>
            {voted === 'up' && (
                <div style={{
                textAlign: 'center', fontSize: 12, color: '#2d8f4e',
                padding: '7px', background: '#f0faf4', borderRadius: 'var(--radius-sm)',
                border: '1px solid #b7dfc7'
                }}>{t('vote_up_thanks')}</div>
            )}
            {voted === 'down' && (
                <div style={{
                textAlign: 'center', fontSize: 12, color: 'var(--text-muted)',
                padding: '7px', background: 'var(--surface-2)', borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border)'
                }}>{t('vote_down_noted')}</div>
            )}
            {!voted && (
                <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={() => handleVote('up')} style={{
                    flex: 1, padding: '7px 0', border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)', background: 'var(--surface-2)', 
                    cursor: 'pointer', fontSize: 15, transition: 'border-color .15s',
                }}>👍</button>
                <button onClick={() => handleVote('down')} style={{
                    flex: 1, padding: '7px 0', border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)', background: 'var(--surface-2)', 
                    cursor: 'pointer', fontSize: 15, transition: 'border-color .15s',
                }}>👎</button>
                </div>
            )}
            </div>
        </div>
        </div>
    )
}