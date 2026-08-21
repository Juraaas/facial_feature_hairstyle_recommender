import { useState, useEffect } from "react"
import { StyleCard } from "./StyleCard"
import { useTranslation } from 'react-i18next'

export function StylesSection({ styles, features, gender}) {
    const { t, i18n } = useTranslation()
    const [page, setPage] = useState(0)
    const isDesktop = window.innerWidth >= 720
    const perPage = isDesktop ? 3 : 1
    const totalPages = Math.ceil(styles.length / perPage)
    const start = page * perPage
    const visible = styles.slice(start , start + perPage)

    useEffect(() => { setPage(0) }, [styles, i18n.language])

    function prev() { setPage(p => Math.max(0, p - 1)) }
    function next() { setPage(p => Math.min(totalPages - 1, p + 1)) }

    return (
        <section style={{ marginBottom: 32 }}>
            {/* nav-header */}
            <div style={{display: 'flex', justifyContent: 'space-between', 
                alignItems: 'center', marginBottom: 20}}>
                <h2 style={{
                    fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 500,
                    color: 'var(--text)', letterSpacing: '.01em',
                }}>{t('section_hairstyles')}</h2>

                {/* page indicator */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)',
                        color: 'var(--text-hint)',
                    }}>{page + 1} / {totalPages}</span>
                    <button
                        onClick={prev}
                        disabled={page === 0}
                        style={navBtn(page === 0)}
                    >←</button>
                    <button
                        onClick={next}
                        disabled={page === totalPages - 1}
                        style={navBtn(page === totalPages - 1)}
                    >→</button>
                </div>
            </div>

            {/* cards */}
            <div className="styles-grid" style={{ animation: 'fadeIn .25s ease' }}
                key={page}>
                {visible.map((style, i) => (
                    <StyleCard
                        key={`${style.name}-${page}-${i}`}
                        style={style}
                        rank={start + i}
                        features={features}
                        gender={gender}
                    />
                ))}
            </div>

            {/* dots */}
            <div style={{display: 'flex', justifyContent: 'center', gap: 6, marginTop: 16}}>
                {Array.from({ length: totalPages }).map((_, i) => (
                    <button
                        key={i}
                        onClick={() => setPage(i)}
                        style={{
                            width: i === page ? 20 : 6, height: 6, borderRadius: 3,
                            border: 'none', cursor: 'pointer',
                            background: i === page
                                ? 'var(--accent)'
                                : 'var(--border)', transition: 'all .25s', padding: 0}}
                    />
                ))}
            </div>
        </section>
    )
}

function navBtn(disabled) {
    return {
        width: 32, height: 32,
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border)',
        background: 'var(--surface)',
        cursor: disabled ? 'default' : 'pointer',
        fontSize: 14,
        color: disabled ? 'var(--text-hint)' : 'var(--text)',
        opacity: disabled ? 0.4 : 1,
        transition: 'all .15s',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    }
}