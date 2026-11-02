/**
 * The server's ceiling on an inbound frame (`_MAX_FRAME_BYTES` in ws/server/serving.py).
 * A frame past it makes the server close the connection.
 */
export const MAX_FRAME_BYTES = 32 * 1024 * 1024;
