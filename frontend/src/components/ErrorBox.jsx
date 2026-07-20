const ERROR_CONFIG = {
    NO_FACE_DETECTED: {
        icon: "🔍",
        title: "No face detected",
        hint: "Make sure your face is clearly visible and well-lit",
    },
    FACE_TOO_SMALL: {
        icon: "📏",
        title: "Move closer to the camera",
        hint: "Your face should fill at least 1/3 of the photo",
    },
    FACE_ROTATED: {
        icon: "↩️",
        title: "Please face the camera directly",
        hint: "Look straight at the lens, not to the side",
    },
    FACE_TILTED: {
        icon: "↕️",
        title: "Head is tilted too much",
        hint: "Try a straight-on photo with your chin level",
    },
    POOR_ALIGNMENT: {
        icon: "⚠️",
        title: "Could not read your face",
        hint: "Try better lighting or a clearer front-facing photo",
    },
    INVALID_IMAGE: {
        icon: "🖼️",
        title: "Could not read the image",
        hint: "Try a JPG or PNG file",
    },
    INTERNAL_ERROR: {
        icon: "⚙️",
        title: "Something went wrong",
        hint: "Please try again with a different photo",
    },
}

export function ErrorBox({error}) {
    if (!error) return null

    const code = typeof error === "object" ? error.code : "INTERNAL_ERROR"
    const msg = typeof error === "object" ? error.message : error
    const config = ERROR_CONFIG[code] || ERROR_CONFIG.INTERNAL_ERROR

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
                {config.title}
            </p>
            <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                {config.hint}
            </p>
            {msg && msg !== config.title && (
                <p style={{
                    fontSize: 11, color: 'var(--text-hint)',
                    marginTop: 8, fontFamily: 'monospace'}}>
                    {msg}
                </p>
            )}
        </div>
    )
}