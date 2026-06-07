import { useState } from "react";
import { sendVote } from '../api/client'

const API_URL = import.meta.env.VITE_API_URL;

export function StyleCard({style, rank, features, gender, onReplace }) {
    const [voted, setVoted] = useState(null)
    const score = Math.max(0, Math.min(100, Math.round(style.score * 100)))
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
        borderRadius: 12,
        overflow: 'hidden',
        border: isTop ? '1.5px solid var(--accent-mid)' : '0.5px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        animation: 'fadeIn .3s ease',
        }}>
        {/* image */}
        <div style={{ position: 'relative' }}>
            {imgPath
            ? <img src={imgPath} alt={style.name}
                style={{ width: '100%', height: 400, objectFit: 'cover', display: 'block' }} />
            : <div style={{ width: '100%', height: 400, background: '#f4f4f4',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 28, color: '#ccc' }}>✂</div>
            }
            {isTop && (
            <div style={{
                position: 'absolute', top: 8, left: 8,
                background: '#185FA5', color: '#E6F1FB',
                fontSize: 10, fontWeight: 500, padding: '2px 8px', borderRadius: 20
            }}>Best match</div>
            )}
            <div style={{
            position: 'absolute', top: 8, right: 8,
            width: 38, height: 30, borderRadius: '50%',
            background: '#fff', border: '0.5px solid #e0e0e0',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 11, fontWeight: 500, color: '#111'
            }}>{score}%</div>
        </div>

        {/* body */}
        <div style={{ padding: 12, display: 'flex', flexDirection: 'column', flex: 1 }}>
            <p style={{ fontSize: 14, fontWeight: 500, marginBottom: 4, color: 'var(--text)'}}>{style.name}</p>

            {/* tags */}
            <div style={{ marginBottom: 8 }}>
            {style.tags?.slice(0, 2).map(t => (
                <span key={t} style={{
                fontSize: 10, padding: '2px 7px', borderRadius: 20,
                background: 'var(--bg)',
                color: 'var(--text-muted)',
                border: '0.5px solid var(--border)', marginRight: 4
                }}>{t}</span>
            ))}
            </div>

            {/* description */}
            {style.description && (
            <p style={{
                fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5,
                marginBottom: 8, paddingBottom: 8,
                borderBottom: '0.5px solid #eee',
                display: '-webkit-box', WebkitLineClamp: 3,
                WebkitBoxOrient: 'vertical', overflow: 'hidden'
            }}>{style.description}</p>
            )}

            {/* why it works */}
            {style.contributions?.length > 0 && (
            <div style={{ marginBottom: 8 }}>
                <p style={{
                fontSize: 10, fontWeight: 500, color: 'var(--text-muted)',
                textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6
                }}>Why it works for you</p>
                {style.contributions.slice(0, 2).map((c, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
                    <span style={{
                    fontSize: 11, color: 'var(--text-muted)', flexShrink: 0,
                    width: 90, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
                    }}>{c.desc}</span>
                    <div style={{ flex: 1, height: 3, borderRadius: 2, background: '#eee' }}>
                    <div style={{
                        width: `${c.percent * 100}%`, height: '100%',
                        borderRadius: 2, background: '#378ADD'
                    }} />
                    </div>
                    <span style={{ fontSize: 11, fontWeight: 500, width: 28, textAlign: 'right', color: 'var(--text)' }}>
                    {Math.round(c.percent * 100)}%
                    </span>
                </div>
                ))}
            </div>
            )}

            {/* negatives */}
            {style.negatives?.length > 0 && (
            <div style={{
                fontSize: 11, padding: '5px 8px',
                borderRadius: 6, marginBottom: 8,
                background: 'var(--bg)',
                borderLeft: '2px solid var(--border)',
                color: 'var(--text-muted)',
            }}>
                ⚠ {style.negatives[0].reason}
            </div>
            )}

            {/* missing */}
            {style.missing?.length > 0 && (
            <div style={{
                fontSize: 11, padding: '5px 8px',
                borderRadius: 6, marginBottom: 8,
                background: 'var(--bg)',
                borderLeft: '2px solid var(--border)',
                color: 'var(--text-muted)',
            }}>
                ℹ {style.missing[0].reason}
            </div>
            )}

            {/* vote */}
            <div style={{ marginTop: 'auto', paddingTop: 8 }}>
            {voted === 'up' && (
                <div style={{
                textAlign: 'center', fontSize: 12, color: '#2d8f4e',
                padding: 6, background: '#f0faf4', borderRadius: 8,
                border: '0.5px solid #b7dfc7'
                }}>👍 Thanks for your feedback!</div>
            )}
            {voted === 'down' && (
                <div style={{
                textAlign: 'center', fontSize: 12, color: '#888',
                padding: 6, background: '#f9f9f9', borderRadius: 8,
                border: '0.5px solid #ddd'
                }}>👎 Noted — showing next suggestion</div>
            )}
            {!voted && (
                <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={() => handleVote('up')} style={{
                    flex: 1, padding: '6px 0', border: '0.5px solid #e0e0e0',
                    borderRadius: 8, background: '#fff', cursor: 'pointer', fontSize: 16
                }}>👍</button>
                <button onClick={() => handleVote('down')} style={{
                    flex: 1, padding: '6px 0', border: '0.5px solid #e0e0e0',
                    borderRadius: 8, background: '#fff', cursor: 'pointer', fontSize: 16
                }}>👎</button>
                </div>
            )}
            </div>
        </div>
        </div>
    )
}