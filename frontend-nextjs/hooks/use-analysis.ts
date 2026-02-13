'use client';

import { useState } from 'react';
import { analyzeMedia } from '@/lib/api/analysis';
import type { AnalysisResult, AnalysisStatus } from '@/types/analysis';

export function useAnalysis() {
    const [status, setStatus] = useState<AnalysisStatus>('idle');
    const [uploadProgress, setUploadProgress] = useState(0);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    async function analyze(file: File, model: string, dataset: string) {
        setStatus('uploading');
        setUploadProgress(0);
        setError(null);
        setResult(null);

        try {
            const res = await analyzeMedia(file, model, dataset, (progress) => {
                setUploadProgress(progress);
                if (progress === 100) setStatus('analyzing');
            });

            setResult(res);
            setStatus('complete');
            setUploadProgress(100);
            return res;
        } catch (err) {
            const msg = err instanceof Error ? err.message : 'Analysis failed. Try again.';
            setError(msg);
            setStatus('error');
            throw err;
        }
    }

    function reset() {
        setResult(null);
        setError(null);
        setStatus('idle');
        setUploadProgress(0);
    }

    return {
        analyze,
        reset,
        status,
        isAnalyzing: status === 'uploading' || status === 'analyzing',
        uploadProgress,
        result,
        error,
    };
}
