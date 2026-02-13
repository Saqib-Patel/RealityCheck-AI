import axios, { AxiosError, AxiosResponse, InternalAxiosRequestConfig } from 'axios';

const baseURL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000';

// Debug log for production troubleshooting
if (typeof window !== 'undefined') {
    console.log('[API Client] Base URL:', baseURL);
}

export const apiClient = axios.create({
    baseURL,
    headers: { 'Content-Type': 'application/json' },
    timeout: 90000, // 90s to handle Render free tier cold starts
});

apiClient.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
        // Log all requests for debugging
        console.log(`[API] ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`);

        // bust cache on GETs
        if (config.method === 'get') {
            config.params = { ...config.params, _t: Date.now() };
        }

        return config;
    },
    (error: AxiosError) => Promise.reject(error)
);

apiClient.interceptors.response.use(
    (response: AxiosResponse) => response,
    (error: AxiosError) => {
        // Always log errors, even in production, for monitoring
        if (error.response) {
            console.error(`[API] ${error.response.status} ${error.config?.url}`, error.response.data);
        } else if (error.request) {
            console.error('[API] No response:', error.message);
        }
        return Promise.reject(error);
    }
);

export const createUploadConfig = (onProgress?: (progress: number) => void) => ({
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e: { loaded: number; total?: number }) => {
        if (onProgress && e.total) {
            onProgress(Math.round((e.loaded * 100) / e.total));
        }
    },
    timeout: 300000, // 5min for big files
});

export default apiClient;
