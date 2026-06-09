import { useState } from "react"
import { sendFeedback } from '../api/client'

export function FeedbackSection({ features, qualityScore, topStyles }) {
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
                    padding: '12px 16px', background: 'var(--surface)',
                    borderRadius: 8, border: '0.5px solid var(--border)',
                    fontSize: 13, color: 'var(--text-muted)',
                    textAlign: 'center'
                }}>
                    ✓ Thanks for your feedback!
                </div>
            </section>
        )
    }

    return (
        <section style={{ marginBottom: 32 }}>
            <h2 style={{ fontSize: 18, fontWeight: 500, marginBottom: 16 }}>
                How accurate were our recommendations?
            </h2>

            {/* stars */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
                {[1, 2, 3, 4, 5].map(n => (
                    <button
                        key={n}
                        onClick={() => setRating(n)}
                        style={{
                            fontSize: 24, background: 'none', border: 'none',
                            cursor: 'pointer', padding: '4px 2px',
                            opacity: rating !== null && n > rating ? 0.3 : 1,
                            transform: rating === n ? 'scale(1.2)' : 'scale(1)',
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
                placeholder="Any comments? (optional)"
                rows={3}
                style={{
                    width: '100%', padding: '10px 12px',
                    border: '0.5px solid var(--border)',
                    borderRadius: 8, background: 'var(--surface)',
                    color: 'var(--text)', fontSize: 13,
                    resize: 'vertical', marginBottom: 12,
                    fontFamily: 'inherit',
                }}
            />

            <button
                onClick={handleSubmit}
                disabled={rating === null || saving}
                className="analyse-btn"
                style={{ opacity: rating === null ? 0.5 : 1 }}
            >
                {saving ? 'Submitting...' : 'Submit feedback'}
            </button>
        </section>
    )
}