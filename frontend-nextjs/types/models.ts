export interface Model {
    name: string;
    description: string;
    accuracy: number;
    speed: 'Fast' | 'Medium' | 'Slow';
    architecture: 'EfficientNet' | 'Xception';
    variant?: 'Standard' | 'Siamese Tuning' | 'Auto-Attention' | 'Auto-Attention + Siamese';
}

export interface Dataset {
    name: string;
    description: string;
    video_count?: number;
    manipulation_methods?: string[];
}

export interface ModelsResponse {
    models: Model[];
    datasets: Dataset[];
}
