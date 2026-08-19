import { useState } from "react"
import { sendFeedback } from '../api/client'
import { useTranslation } from 'react-i18next'

export function FeedbackSection({ features, qualityScore, topStyles }) {
    const { t } = useTranslation()
    const [rating, setRating] = useState(null)
    const [comment, setComment] = useState('')
    const [saved, setSaved] = useState(false)
    const [saving, setSaving] = useState(false)

    async function handleSubmit() {
        if (rating === null) return
        setSaving(true)
        try {
            await sendFeedback(features, qualityScore, topStyles, rating, comment)
            setSaved(true)
        } catch (e) {
            console.error(e)
        } finally {
            setSaving(false)
        }
    }

    if (saved) {
        return (
            <section style={{ marginBottom: 32 }}>
                <div style={{
                    padding: '14px 16px', background: 'var(--surface)',
                    borderRadius: 'var(--radius-md)', border: '1px solid var(--border)',
                    fontSize: 13, color: 'var(--text-muted)',
                    textAlign: 'center', fontWeight: 300,
                }}>
                    {t('feedback_saved')}
                </div>
            </section>
        )
    }

    return (
        <section style={{ marginBottom: 40 }}>
            <h2 className="section-title">{t('section_feedback')}</h2>
            <div style={{background: 'var(--surface)', borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border)', padding: '20px'}}>
                {/* stars */}
                <div style={{ display: 'flex', gap: 6, marginBottom: 16 }}>
                    {[1, 2, 3, 4, 5].map(n => (
                        <button
                            key={n}
                            onClick={() => setRating(n)}
                            style={{
                                fontSize: 22, background: 'none', border: 'none',
                                cursor: 'pointer', padding: '4px 2px',
                                opacity: rating !== null && n > rating ? 0.25 : 1,
                                transform: rating === n ? 'scale(1.25)' : 'scale(1)',
                                transition: 'all .15s',
                            }}
                        >
                            ⭐
                        </button>
                    ))}
                </div>

                {/* comment */}
                <textarea
                    value={comment}
                    onChange={e => setComment(e.target.value)}
                    placeholder={t('feedback_comment_placeholder')}
                    rows={3}
                    style={{
                        width: '100%', padding: '10px 12px',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-md)', background: 'var(--surface-2)',
                        color: 'var(--text)', fontSize: 13,
                        resize: 'vertical', marginBottom: 14,
                        outline: 'none', lineHeight: 1.5,
                    }}
                />

                <button
                    onClick={handleSubmit}
                    disabled={rating === null || saving}
                    className="analyse-btn"
                    style={{ opacity: rating === null ? 0.45 : 1 }}
                >
                    {saving ? t('feedback_submitting') : t('feedback_submit')}
                </button>
            </div>
        </section>
    )
}