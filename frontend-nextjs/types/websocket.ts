import type { AnalysisProgress, AnalysisResult, AnalysisError } from './analysis';

export interface WebSocketEvents {
    connect: () => void;
    disconnect: (reason: string) => void;
    analysis_progress: (data: AnalysisProgress) => void;
    analysis_complete: (data: AnalysisResult) => void;
    analysis_error: (data: AnalysisError) => void;
    join_room: (data: { room: string }) => void;
    leave_room: (data: { room: string }) => void;
}

export type WebSocketEventType = keyof WebSocketEvents;

export interface WebSocketState {
    isConnected: boolean;
    socketId: string | null;
    error: string | null;
}
