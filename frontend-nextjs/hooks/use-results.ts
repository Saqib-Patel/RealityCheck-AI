'use client';

import { useState, useEffect } from 'react';
import { getResults, deleteResult as apiDelete } from '@/lib/api/results';
import type { AnalysisResult } from '@/types/analysis';

export function useResults() {
    const [results, setResults] = useState<AnalysisResult[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    async function fetchResults() {
        try {
            setIsLoading(true);
            setError(null);
            setResults(await getResults());
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch results');
        } finally {
            setIsLoading(false);
        }
    }

    useEffect(() => { fetchResults(); }, []);

    async function deleteResult(id: string) {
        await apiDelete(id);
        setResults(prev => prev.filter(r => r.id !== id));
    }

    return { results, isLoading, error, deleteResult, refresh: fetchResults };
}
