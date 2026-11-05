import { Component } from 'react'

export class ErrorBoundary extends Component {
  state = { hasError: false }
  static getDerivedStateFromError() { return { hasError: true } }
  componentDidCatch(error, info) { console.error(error, info) }
  render() {
    if (!this.state.hasError) return this.props.children
    return (
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', gap: 12, padding: 24, textAlign: 'center' }}>
        <h2 style={{ fontFamily: 'var(--font-display)' }}>Coś poszło nie tak · Something went wrong</h2>
        <button onClick={() => window.location.reload()} className="analyse-btn" style={{ maxWidth: 220 }}>
          Odśwież · Reload
        </button>
      </div>
    )
  }
}