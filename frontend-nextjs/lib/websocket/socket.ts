'use client';

import { io, Socket } from 'socket.io-client';
import { WS_EVENTS } from '../api/endpoints';

let socket: Socket | null = null;

export interface SocketConfig {
    url?: string;
    reconnection?: boolean;
    reconnectionAttempts?: number;
    reconnectionDelay?: number;
    timeout?: number;
}

export function initializeSocket(config?: SocketConfig): Socket {
    if (socket?.connected) return socket;

    const url = config?.url || process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:5000';

    socket = io(url, {
        transports: ['websocket', 'polling'],
        reconnection: config?.reconnection ?? true,
        reconnectionAttempts: config?.reconnectionAttempts ?? 5,
        reconnectionDelay: config?.reconnectionDelay ?? 1000,
        timeout: config?.timeout ?? 20000,
        autoConnect: true,
    });

    // Development-only logging
    if (process.env.NODE_ENV === 'development') {
        socket.on(WS_EVENTS.CONNECT, () => console.log('[WS] Connected:', socket!.id));
        socket.on(WS_EVENTS.DISCONNECT, (reason) => console.log('[WS] Disconnected:', reason));
    }

    // Always log errors for monitoring
    socket.on('connect_error', (err) => console.error('[WS] Error:', err.message));

    return socket;
}

export const getSocket = () => socket;

export function disconnectSocket() {
    if (socket) {
        socket.disconnect();
        socket = null;
    }
}

export const isSocketConnected = () => socket?.connected ?? false;

export function joinRoom(roomId: string) {
    if (socket?.connected) socket.emit(WS_EVENTS.JOIN_ROOM, { room: roomId });
}

export function leaveRoom(roomId: string) {
    if (socket?.connected) socket.emit(WS_EVENTS.LEAVE_ROOM, { room: roomId });
}

export default { initializeSocket, getSocket, disconnectSocket, isSocketConnected, joinRoom, leaveRoom };
