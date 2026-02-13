export const APP_NAME = 'DeepFake Detection Hub';
export const APP_VERSION = '2.0.0';
export const APP_DESCRIPTION = 'AI-powered deepfake detection using state-of-the-art deep learning models';

export const ROUTES = {
    HOME: '/',
    ANALYZE: '/analyze',
    HOW_IT_WORKS: '/how-it-works',
    HISTORY: '/history',
    RESULT_DETAIL: (id: string) => `/history/${id}`,
} as const;

export const FILE_CONSTRAINTS = {
    MAX_SIZE: 5 * 1024 * 1024, // 5MB
    ALLOWED_IMAGE_TYPES: ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'],
    ALLOWED_VIDEO_TYPES: ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime'],
} as const;

export const DEFAULT_MODEL = 'EfficientNetAutoAttB4';
export const DEFAULT_DATASET = 'DFDC';

export const PREDICTION_COLORS = {
    real: {
        bg: 'bg-success-400',
        text: 'text-success-400',
        border: 'border-success-400',
        gradient: 'from-success-400 to-accent-cyan',
    },
    fake: {
        bg: 'bg-danger-400',
        text: 'text-danger-400',
        border: 'border-danger-400',
        gradient: 'from-danger-400 to-neon-orange',
    },
} as const;

export const MODELS = [
    {
        name: 'EfficientNetB4',
        description: 'Standard EfficientNet-B4 architecture',
        accuracy: 90,
        speed: 'Fast' as const,
    },
    {
        name: 'EfficientNetB4ST',
        description: 'EfficientNet-B4 with Siamese Tuning',
        accuracy: 92,
        speed: 'Fast' as const,
    },
    {
        name: 'EfficientNetAutoAttB4',
        description: 'EfficientNet-B4 with Auto-Attention (Recommended)',
        accuracy: 93,
        speed: 'Medium' as const,
    },
    {
        name: 'EfficientNetAutoAttB4ST',
        description: 'EfficientNet-B4 with Auto-Attention + Siamese Tuning',
        accuracy: 94,
        speed: 'Medium' as const,
    },
] as const;

export const DATASETS = [
    {
        name: 'DFDC',
        description: 'DeepFake Detection Challenge (Facebook)',
        videoCount: 100000,
    },
    {
        name: 'FFPP',
        description: 'FaceForensics++',
        videoCount: 1000,
    },
] as const;

export const SOCIAL_LINKS = {
    github: 'https://github.com/Saqib-Patel',
    twitter: 'https://x.com/patel_saqib26',
    linkedin: 'https://www.linkedin.com/in/mohammedsaqibpatel/',
} as const;
