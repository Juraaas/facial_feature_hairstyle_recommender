import { useState } from "react"
import { sendFeedback } from '../api/client'
import { useTranslation } from 'react-i18next'

import { Star } from 'lucide-react'

function StarRating({ rating, onChange }) {
  const [hovered, setHovered] = useState(null)
  const display = hovered ?? rating ?? 0

  return (
    <div style={{ display: 'flex', gap: 4, marginBottom: 16 }}>
      {[1, 2, 3, 4, 5].map(n => {
        const full = display >= n
        const half = !full && display >= n - 0.5

        return (
          <div
            key={n}
            style={{ position: 'relative', width: 28, height: 28, cursor: 'pointer' }}
            onMouseLeave={() => setHovered(null)}
          >
            {/* left half */}
            <div
              style={{ position: 'absolute', left: 0, top: 0, width: '50%', height: '100%', zIndex: 1 }}
              onMouseEnter={() => setHovered(n - 0.5)}
              onClick={() => onChange(n - 0.5)}
            />
            {/* right half */}
            <div
              style={{ position: 'absolute', right: 0, top: 0, width: '50%', height: '100%', zIndex: 1 }}
              onMouseEnter={() => setHovered(n)}
              onClick={() => onChange(n)}
            />
            {/* icon */}
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {half ? (
                <div style={{ position: 'relative', width: 22, height: 22 }}>
                  <Star size={22} color="var(--border)" fill="var(--border)" strokeWidth={0} />
                  <div style={{ position: 'absolute', top: 0, left: 0, width: '50%', overflow: 'hidden' }}>
                    <Star size={22} color="var(--accent)" fill="var(--accent)" strokeWidth={0} />
                  </div>
                </div>
              ) : (
                <Star
                  size={22}
                  color={full ? 'var(--accent)' : 'var(--border)'}
                  fill={full ? 'var(--accent)' : 'var(--border)'}
                  strokeWidth={0}
                />
              )}
            </div>
          </div>
        )
      })}
      {rating !== null && (
        <span style={{
          fontSize: 12, color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)', marginLeft: 6,
          alignSelf: 'center',
        }}>
          {rating}/5
        </span>
      )}
    </div>
  )
}


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
                <StarRating rating={rating} onChange={setRating} />
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