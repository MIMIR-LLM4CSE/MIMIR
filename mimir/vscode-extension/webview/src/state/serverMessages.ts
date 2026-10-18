// The server's messages, routed by type to whoever holds the state they update.

import type { ServerMessage } from "../types";

export type ServerMessageOf<T extends ServerMessage["type"]> = Extract<ServerMessage, { type: T }>;

/** One handler per message type a part of the panel answers. */
export type ServerHandlers = {
  [T in ServerMessage["type"]]?: (msg: ServerMessageOf<T>) => void;
};

/** Run every handler of *msg*'s type across *tables*; true when one existed. */
export function routeServerMessage(msg: ServerMessage, tables: ServerHandlers[]): boolean {
  let handled = false;
  for (const table of tables) {
    const handler = table[msg.type] as ((m: ServerMessage) => void) | undefined;
    if (handler) {
      handler(msg);
      handled = true;
    }
  }
  return handled;
}
