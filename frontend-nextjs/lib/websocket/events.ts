import type { Socket } from 'socket.io-client';
import { WS_EVENTS } from '../api/endpoints';
import type { AnalysisProgress, AnalysisResult } from '@/types/analysis';

export interface WebSocketEventHandlers {
    onProgress?: (data: AnalysisProgress) => void;
    onComplete?: (data: AnalysisResult) => void;
    onError?: (error: { analysis_id: string; error: string }) => void;
    onConnect?: () => void;
    onDisconnect?: () => void;
}

const eventMap: Record<string, keyof WebSocketEventHandlers> = {
    [WS_EVENTS.CONNECT]: 'onConnect',
    [WS_EVENTS.DISCONNECT]: 'onDisconnect',
    [WS_EVENTS.ANALYSIS_PROGRESS]: 'onProgress',
    [WS_EVENTS.ANALYSIS_COMPLETE]: 'onComplete',
    [WS_EVENTS.ANALYSIS_ERROR]: 'onError',
};

export function registerEventHandlers(socket: Socket, handlers: WebSocketEventHandlers) {
    for (const [event, key] of Object.entries(eventMap)) {
        const handler = handlers[key];
        if (handler) socket.on(event, handler as (...args: unknown[]) => void);
    }
}

export function unregisterEventHandlers(socket: Socket, handlers: WebSocketEventHandlers) {
    for (const [event, key] of Object.entries(eventMap)) {
        const handler = handlers[key];
        if (handler) socket.off(event, handler as (...args: unknown[]) => void);
    }
}

export function onceEvent<T>(socket: Socket, event: string, cb: (data: T) => void) {
    socket.once(event, cb);
}

export function removeAllListeners(socket: Socket, event?: string) {
    event ? socket.removeAllListeners(event) : socket.removeAllListeners();
}
