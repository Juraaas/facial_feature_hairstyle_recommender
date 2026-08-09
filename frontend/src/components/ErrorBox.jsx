import { useTranslation } from "react-i18next"

const ERROR_CONFIG = {
  NO_FACE_DETECTED: {
    icon: "🔍",
    titleKey: "error_no_face",
    hintKey: "error_no_face_hint",
  },

  FACE_TOO_SMALL: {
    icon: "📏",
    titleKey: "error_too_small",
    hintKey: "error_too_small_hint",
  },

  FACE_ROTATED: {
    icon: "↩️",
    titleKey: "error_rotated",
    hintKey: "error_rotated_hint",
  },

  FACE_TILTED: {
    icon: "↕️",
    titleKey: "error_tilted",
    hintKey: "error_tilted_hint",
  },

  POOR_ALIGNMENT: {
    icon: "⚠️",
    titleKey: "error_alignment",
    hintKey: "error_alignment_hint",
  },

  INVALID_IMAGE: {
    icon: "🖼️",
    titleKey: "error_invalid_image",
    hintKey: "error_invalid_image_hint",
  },

  INTERNAL_ERROR: {
    icon: "⚙️",
    titleKey: "error_internal",
    hintKey: "error_internal_hint",
  },
}

export function ErrorBox({error}) {
    const { t } = useTranslation()
    if (!error) return null

    const code = typeof error === "object" ? error.code : "INTERNAL_ERROR"
    const msg = typeof error === "object" ? error.message : error
    const config = ERROR_CONFIG[code] || ERROR_CONFIG.INTERNAL_ERROR

    const title = t(config.titleKey)
    const hint = t(config.hintKey)

    return (
        <div style={{
            background: 'var(--surface)',
            border: '0.5px solid var(--border)',
            borderRadius: 12,
            padding: '16px 20px',
            marginBottom: 16,
        }}>
            <div style={{ fontSize: 24, marginBottom: 8 }}>
                {config.icon}
            </div>
            <p style={{
                fontSize: 14, fontWeight: 500, color: 'var(--text)', marginBottom: 4}}>
                {title}
            </p>
            <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                {hint}
            </p>
            {msg && msg !== title && (
                <p style={{
                    fontSize: 11, color: 'var(--text-hint)',
                    marginTop: 8, fontFamily: 'monospace'}}>
                    {msg}
                </p>
            )}
        </div>
    )
}