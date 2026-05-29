import { useState } from "react"
import { StyleCard } from "./StyleCard"

export function StylesSection({ styles, features, gender}) {
    const [displayed, setDisplayed] = useState(styles.slice(0, 3))
    const [queue, setQueue] = useState(styles.slice(3))

    function handleReplace(idx) {
        if (!queue.length) return
        const next = queue[0]
        const newQueue = queue.slice(1)
        const newDisp = [...displayed]
        newDisp[idx] = next
        setDisplayed(newDisp)
        setQueue(newQueue)
    }

    return (
        <section style={{ marginBottom: 32 }}>
        <h2 style={{ fontSize: 18, fontWeight: 500, marginBottom: 20 }}>Top hairstyles</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
            {displayed.map((style, i) => (
            <StyleCard
                key={`${i}-${style.name}`}
                style={style}
                rank={i}
                features={features}
                gender={gender}
                onReplace={handleReplace}
            />
            ))}
        </div>
        </section>
  )
}