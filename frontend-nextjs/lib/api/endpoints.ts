export const API_ENDPOINTS = {
    MODELS: '/api/v1/models',
    ANALYZE_IMAGE: '/api/v1/analyze/image',
    ANALYZE_VIDEO: '/api/v1/analyze/video',
    ANALYZE_BATCH: '/api/v1/analyze/batch',
    COMPARE: '/api/v1/compare',
    RESULTS: '/api/v1/results',
    RESULT_BY_ID: (id: string) => `/api/v1/results/${id}`,
    HEALTH: '/health/',
} as const;

export const WS_EVENTS = {
    CONNECT: 'connect',
    DISCONNECT: 'disconnect',
    ANALYSIS_PROGRESS: 'analysis_progress',
    ANALYSIS_COMPLETE: 'analysis_complete',
    ANALYSIS_ERROR: 'analysis_error',
    JOIN_ROOM: 'join_room',
    LEAVE_ROOM: 'leave_room',
} as const;

export type ApiEndpoint = typeof API_ENDPOINTS;
export type WsEvent = typeof WS_EVENTS;
