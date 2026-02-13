import { apiClient } from './client';
import { API_ENDPOINTS } from './endpoints';
import type { AnalysisResult } from '@/types/analysis';

// Maps the backend's field names to what the frontend expects
function mapResult(data: Record<string, unknown>): AnalysisResult {
    return {
        id: (data.analysis_id as string) || (data.id as string) || '',
        filename: (data.filename as string) || (data.original_filename as string) || 'Unknown',
        file_type: ((data.type as string) || 'image') as 'image' | 'video',
        file_size: (data.file_size as number) || 0,
        prediction: (data.verdict as string) === 'fake' || (data.is_fake as boolean) ? 'fake' : 'real',
        confidence: (data.confidence as number) || 0,
        model_name: (data.model as string) || (data.model_name as string) || 'Unknown',
        model_architecture: (data.model as string) || (data.model_architecture as string) || 'EfficientNet',
        dataset: (data.dataset as string) || 'DFDC',
        processing_time: (data.processing_time as number) || 0,
        faces_detected: (data.faces_detected as number) || 0,
        created_at: (data.timestamp as string) || (data.created_at as string) || new Date().toISOString(),
    };
}

export async function getResults(): Promise<AnalysisResult[]> {
    try {
        const { data } = await apiClient.get(API_ENDPOINTS.RESULTS);
        const results = Array.isArray(data) ? data : (data?.results || []);
        return results.map((r: Record<string, unknown>) => mapResult(r));
    } catch (err) {
        console.error('Failed to fetch results:', err);
        return [];
    }
}

export async function getResultById(id: string) {
    const { data } = await apiClient.get(API_ENDPOINTS.RESULT_BY_ID(id));
    return mapResult(data);
}

export async function deleteResult(id: string) {
    await apiClient.delete(API_ENDPOINTS.RESULT_BY_ID(id));
}

export async function deleteMultipleResults(ids: string[]) {
    await Promise.all(ids.map(id => deleteResult(id)));
}

export async function getResultsStats() {
    const results = await getResults();
    return {
        total: results.length,
        real: results.filter(r => r.prediction === 'real').length,
        fake: results.filter(r => r.prediction === 'fake').length,
    };
}
