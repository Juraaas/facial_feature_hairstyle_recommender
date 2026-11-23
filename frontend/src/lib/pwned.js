export const MIN_PASSWORD = 10

export async function isPwned(password) {
  try {
    const data = new TextEncoder().encode(password)
    const digest = await crypto.subtle.digest('SHA-1', data)
    const hex = Array.from(new Uint8Array(digest))
      .map(b => b.toString(16).padStart(2, '0')).join('').toUpperCase()
    const prefix = hex.slice(0, 5)
    const suffix = hex.slice(5)
    const res = await fetch(`https://api.pwnedpasswords.com/range/${prefix}`, {
      headers: { 'Add-Padding': 'true' },
    })
    if (!res.ok) return false
    const text = await res.text()
    return text.split('\n').some(line => {
      const [hash, count] = line.trim().split(':')
      return hash === suffix && parseInt(count, 10) > 0
    })
  } catch {
    return false
  }
}