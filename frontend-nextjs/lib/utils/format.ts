import { format, formatDistanceToNow } from 'date-fns';

function safeDate(date: string | Date | undefined | null): Date | null {
    if (!date) return null;
    try {
        const d = typeof date === 'string' ? new Date(date) : date;
        return isNaN(d.getTime()) ? null : d;
    } catch {
        return null;
    }
}

export function formatDate(date: string | Date | undefined | null, fmt = 'PPP') {
    const d = safeDate(date);
    if (!d) return 'N/A';
    try { return format(d, fmt); } catch { return 'N/A'; }
}

export function formatRelativeTime(date: string | Date | undefined | null) {
    const d = safeDate(date);
    if (!d) return 'recently';
    try { return formatDistanceToNow(d, { addSuffix: true }); } catch { return 'recently'; }
}

export function formatFileSize(bytes: number | undefined | null) {
    if (bytes == null || isNaN(bytes) || bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

export const formatConfidence = (v: number | undefined | null) =>
    v == null || isNaN(v) ? '0.0%' : `${(v * 100).toFixed(1)}%`;

export function formatProcessingTime(seconds: number | undefined | null) {
    if (seconds == null || isNaN(seconds)) return '0.0s';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
}

export const formatNumber = (n: number | undefined | null) =>
    n == null || isNaN(n) ? '0' : n.toLocaleString();

export function formatDuration(seconds: number | undefined | null) {
    if (seconds == null || isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}
