export interface AnalysisResult {
    id: string;
    filename: string;
    file_type: 'image' | 'video';
    file_size: number;
    prediction: 'real' | 'fake';
    confidence: number;
    model_name: string;
    model_architecture: string;
    dataset: string;
    processing_time: number;
    faces_detected: number;
    frame_predictions?: FramePrediction[];
    thumbnail_url?: string;
    video_url?: string;
    created_at: string;
    metadata?: {
        video_duration?: number;
        video_fps?: number;
        total_frames?: number;
        resolution?: {
            width: number;
            height: number;
        };
    };
}

export interface FramePrediction {
    frame: number;
    timestamp: number;
    confidence: number;
    prediction: 'real' | 'fake';
}

export interface AnalysisProgress {
    analysis_id: string;
    progress: number;
    current_frame: number;
    total_frames: number;
    status: string;
    eta?: number;
}

export interface BatchAnalysisResult {
    results: AnalysisResult[];
    total_files: number;
    successful: number;
    failed: number;
    errors?: Array<{
        filename: string;
        error: string;
    }>;
}

export interface ComparisonResult {
    file: {
        filename: string;
        file_type: 'image' | 'video';
    };
    comparisons: Array<{
        model_name: string;
        prediction: 'real' | 'fake';
        confidence: number;
        processing_time: number;
    }>;
    consensus: {
        prediction: 'real' | 'fake';
        agreement_percentage: number;
    };
}

export interface AnalysisError {
    analysis_id: string;
    error: string;
    timestamp: string;
}

export type AnalysisStatus = 'idle' | 'uploading' | 'analyzing' | 'complete' | 'error';
