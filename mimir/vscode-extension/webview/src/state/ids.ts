/** A fresh id for a message, a block or a transcript epoch. */
export function makeId(): string {
  return crypto.randomUUID();
}
