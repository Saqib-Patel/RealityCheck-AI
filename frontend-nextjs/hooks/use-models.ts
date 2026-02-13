'use client';

import { useState, useEffect } from 'react';
import { getModels } from '@/lib/api/models';
import type { Model, Dataset } from '@/types/models';

export function useModels() {
    const [models, setModels] = useState<Model[]>([]);
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    async function fetchModels() {
        try {
            setIsLoading(true);
            setError(null);
            const data = await getModels();
            setModels(data.models);
            setDatasets(data.datasets);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch models');
        } finally {
            setIsLoading(false);
        }
    }

    useEffect(() => { fetchModels(); }, []);

    return { models, datasets, isLoading, error, refetch: fetchModels };
}
