'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { initializeSocket, disconnectSocket } from '@/lib/websocket/socket';
import { registerEventHandlers, unregisterEventHandlers } from '@/lib/websocket/events';
import type { Socket } from 'socket.io-client';
import type { WebSocketEventHandlers } from '@/lib/websocket/events';

export function useWebSocket(handlers?: WebSocketEventHandlers) {
    const [socket, setSocket] = useState<Socket | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const handlersRef = useRef(handlers);

    useEffect(() => { handlersRef.current = handlers; }, [handlers]);

    useEffect(() => {
        const s = initializeSocket();
        setSocket(s);

        const onConnect = () => { setIsConnected(true); handlersRef.current?.onConnect?.(); };
        const onDisconnect = () => { setIsConnected(false); handlersRef.current?.onDisconnect?.(); };

        s.on('connect', onConnect);
        s.on('disconnect', onDisconnect);
        setIsConnected(s.connected);

        if (handlersRef.current) registerEventHandlers(s, handlersRef.current);

        return () => {
            s.off('connect', onConnect);
            s.off('disconnect', onDisconnect);
            if (handlersRef.current) unregisterEventHandlers(s, handlersRef.current);
            disconnectSocket();
        };
    }, []);

    const emit = useCallback((event: string, data?: unknown) => {
        if (socket?.connected) socket.emit(event, data);
    }, [socket]);

    const reconnect = useCallback(() => {
        if (socket && !socket.connected) socket.connect();
    }, [socket]);

    return { socket, isConnected, emit, reconnect };
}
