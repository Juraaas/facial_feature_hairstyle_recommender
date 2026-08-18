import { useTranslation } from 'react-i18next'

export function FaceAnalysis({ analysis }) {
    const { t } = useTranslation()

    if (!analysis?.length) {
        return (
            <section style={{ marginBottom: 32 }}>
                <h2 className="section-title">{t('section_face_analysis')}</h2>
                <p style={{ fontSize: 13, color: 'var(--text-muted)', fontWeight: 300 }}>
                    {t('face_balanced')}
                </p>
            </section>
        )
    }

    return (
        <section style={{ marginBottom: 32 }}>
            <h2 className="section-title">{t('section_face_analysis')}</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {analysis.map((exp, i) => (
                    <div key={i} style={{
                        fontSize: 13, lineHeight: 1.6, padding: '8px 12px',
                        borderLeft: '2px solid var(--accent)', color: 'var(--text)',
                        background: 'var(--surface)', borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
                        fontWeight: 300,
                    }}>{exp}</div>
                ))}
            </div>
        </section>
    )
}