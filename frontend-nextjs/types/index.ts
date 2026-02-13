export * from './models';
export * from './analysis';
export * from './websocket';

// Common types
export interface ApiResponse<T = unknown> {
    success: boolean;
    data?: T;
    error?: string;
    message?: string;
}

export interface PaginationParams {
    page: number;
    limit: number;
    sort?: 'asc' | 'desc';
    sortBy?: string;
}

export interface FilterParams {
    model?: string;
    dataset?: string;
    prediction?: 'real' | 'fake';
    dateFrom?: string;
    dateTo?: string;
    search?: string;
}
