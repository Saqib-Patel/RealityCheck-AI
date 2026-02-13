import { apiClient } from './client';
import { API_ENDPOINTS } from './endpoints';
import type { Model, Dataset, ModelsResponse } from '@/types/models';

// Backend uses `id` + decimal accuracy, frontend wants `name` + integer accuracy
function mapModel(data: Record<string, unknown>): Model {
    const id = (data.id as string) || '';
    const speed = (data.speed as string) || 'Medium';

    return {
        name: (data.id as string) || (data.name as string) || 'Unknown',
        description: (data.description as string) || '',
        accuracy: typeof data.accuracy === 'object' && data.accuracy !== null
            ? Math.round(((data.accuracy as Record<string, number>).DFDC || 0) * 100)
            : (data.accuracy as number) || 0,
        speed: (speed.charAt(0).toUpperCase() + speed.slice(1)) as 'Fast' | 'Medium' | 'Slow',
        architecture: 'EfficientNet' as const,
        variant: id.includes('AutoAtt')
            ? (id.includes('ST') ? 'Auto-Attention + Siamese' : 'Auto-Attention')
            : (id.includes('ST') ? 'Siamese Tuning' : 'Standard'),
    };
}

function mapDataset(data: Record<string, unknown>): Dataset {
    return {
        name: (data.id as string) || (data.name as string) || 'Unknown',
        description: (data.description as string) || '',
    };
}

export async function getModels(): Promise<ModelsResponse> {
    try {
        const { data } = await apiClient.get(API_ENDPOINTS.MODELS);
        return {
            models: Array.isArray(data?.models) ? data.models.map(mapModel) : [],
            datasets: Array.isArray(data?.datasets) ? data.datasets.map(mapDataset) : [],
        };
    } catch (err) {
        console.error('Failed to fetch models:', err);
        return { models: [], datasets: [] };
    }
}

export async function getModelByName(name: string) {
    const { models } = await getModels();
    return models.find(m => m.name === name) || null;
}

export async function getDatasetByName(name: string) {
    const { datasets } = await getModels();
    return datasets.find(d => d.name === name) || null;
}

export async function checkModelsHealth() {
    try {
        await getModels();
        return true;
    } catch {
        return false;
    }
}
