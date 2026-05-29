export function FaceAnalysis ({ analysis }) {
    if (!analysis?.length) {
        return (
            <section style={{ marginBottom: 32}}>
                <h2 style={{ fontSize: 18, fontWeight: 500, marginBottom: 16}}>Face analysis</h2>
                <p style={{ fontSize: 13, color: '#888'}}>
                    Proportions are well balanced - most styles will suit you.
                </p>
            </section>
        )
    }
    return (
    <section style={{ marginBottom: 32 }}>
      <h2 style={{ fontSize: 18, fontWeight: 500, marginBottom: 16 }}>Face analysis</h2>
      {analysis.map((exp, i) => (
        <div key={i} style={{
          fontSize: 13, padding: '6px 0 6px 12px',
          borderLeft: '2.5px solid #378ADD',
          marginBottom: 8, lineHeight: 1.5, color: '#333'
        }}>
          {exp}
        </div>
      ))}
    </section>
  )
}