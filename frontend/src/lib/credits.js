export const plural = (n, one, few, many) => {
  if (n === 1) return one
  const m10 = n % 10, m100 = n % 100
  return m10 >= 2 && m10 <= 4 && !(m100 >= 12 && m100 <= 14) ? few : many
}

export const creditWord = (n, pl) =>
  pl ? plural(n, 'kredyt', 'kredyty', 'kredytów') : (n === 1 ? 'credit' : 'credits')