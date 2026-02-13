import { apiClient, createUploadConfig } from './client';
import { API_ENDPOINTS } from './endpoints';
import type { AnalysisResult, BatchAnalysisResult, ComparisonResult } from '@/types/analysis';

// Backend returns different field names — this maps them to our frontend types
function mapResponse(data: Record<string, unknown>, file: File): AnalysisResult {
    if (data.status === 'error' || data.error) {
        throw new Error((data.error as string) || 'Analysis failed');
    }

    const type = (data.type as string) || (file.type.startsWith('video/') ? 'video' : 'image');
    const verdict = data.verdict as string | undefined;
    const isFake = data.is_fake as boolean | undefined;

    return {
        id: (data.analysis_id as string) || crypto.randomUUID(),
        filename: file.name,
        file_type: type as 'image' | 'video',
        file_size: file.size,
        prediction: (verdict === 'real' || verdict === 'fake') ? verdict : (isFake ? 'fake' : 'real'),
        confidence: (data.confidence as number) || 0,
        model_name: (data.model as string) || 'Unknown',
        model_architecture: (data.model as string) || 'EfficientNet',
        dataset: (data.dataset as string) || 'DFDC',
        processing_time: (data.processing_time as number) || 0,
        faces_detected: (data.faces_detected as number) || (data.frames_with_faces as number) || 0,
        frame_predictions: data.frame_details
            ? (data.frame_details as Array<{ frame_index: number; confidence: number; is_fake: boolean }>).map((f, i) => ({
                frame: f.frame_index || i,
                timestamp: i * 0.04,
                confidence: f.confidence,
                prediction: f.is_fake ? 'fake' as const : 'real' as const,
            }))
            : undefined,
        created_at: (data.timestamp as string) || new Date().toISOString(),
        metadata: {
            video_duration: data.frames_analyzed ? ((data.frames_analyzed as number) / 25) : undefined,
            total_frames: data.frames_analyzed as number | undefined,
        },
    };
}

// Helper to wait between polls
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Poll for analysis result until completed or failed
async function pollForResult(analysisId: string, file: File, maxAttempts = 120): Promise<AnalysisResult> {
    console.log(`[Analysis] Polling for result: ${analysisId}`);
    
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const { data, status } = await apiClient.get<Record<string, unknown>>(
            API_ENDPOINTS.RESULT_BY_ID(analysisId),
            { validateStatus: () => true } // Don't throw on non-2xx
        );
        
        console.log(`[Analysis] Poll attempt ${attempt + 1}: status=${data.status}`);
        
        if (status === 200 && data.status === 'completed') {
            return mapResponse(data, file);
        }
        
        if (status === 500 || data.status === 'failed') {
            throw new Error((data.error as string) || 'Analysis failed on server');
        }
        
        // Still processing - wait and retry (2s interval)
        await sleep(2000);
    }
    
    throw new Error('Analysis timed out. Please try again.');
}

async function postAnalysis(endpoint: string, file: File, model: string, dataset: string, onProgress?: (n: number) => void) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    formData.append('dataset', dataset);

    // Upload file - backend returns 202 with analysis_id
    const { data } = await apiClient.post<Record<string, unknown>>(endpoint, formData, createUploadConfig(onProgress));
    
    const analysisId = data.analysis_id as string;
    
    // If backend returned immediate result (status: completed), use it
    if (data.status === 'completed') {
        return mapResponse(data, file);
    }
    
    // Otherwise poll for result
    if (!analysisId) {
        throw new Error('No analysis_id returned from server');
    }
    
    return pollForResult(analysisId, file);
}

export const analyzeImage = (file: File, model: string, dataset: string, onProgress?: (n: number) => void) =>
    postAnalysis(API_ENDPOINTS.ANALYZE_IMAGE, file, model, dataset, onProgress);

export const analyzeVideo = (file: File, model: string, dataset: string, onProgress?: (n: number) => void) =>
    postAnalysis(API_ENDPOINTS.ANALYZE_VIDEO, file, model, dataset, onProgress);

export const analyzeMedia = (file: File, model: string, dataset: string, onProgress?: (n: number) => void) =>
    file.type.startsWith('video/')
        ? analyzeVideo(file, model, dataset, onProgress)
        : analyzeImage(file, model, dataset, onProgress);

export async function analyzeBatch(files: File[], model: string, dataset: string, onProgress?: (n: number) => void) {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    formData.append('model', model);
    formData.append('dataset', dataset);

    const { data } = await apiClient.post<BatchAnalysisResult>(API_ENDPOINTS.ANALYZE_BATCH, formData, createUploadConfig(onProgress));
    return data;
}

export async function compareModels(file: File, models: string[], dataset: string, onProgress?: (n: number) => void) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('models', JSON.stringify(models));
    formData.append('dataset', dataset);

    const { data } = await apiClient.post<ComparisonResult>(API_ENDPOINTS.COMPARE, formData, createUploadConfig(onProgress));
    return data;
}
