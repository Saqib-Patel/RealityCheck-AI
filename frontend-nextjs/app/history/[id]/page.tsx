'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Download, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { AnalysisResultCard } from '@/components/features/analysis-result-card';
import { getResultById, deleteResult } from '@/lib/api/results';
import { downloadJSON } from '@/lib/utils/download';
import { ROUTES } from '@/lib/constants';
import type { AnalysisResult } from '@/types/analysis';

export default function ResultDetailPage() {
    const params = useParams();
    const router = useRouter();
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchResult = async () => {
            try {
                setIsLoading(true);
                const data = await getResultById(params.id as string);
                setResult(data);
            } catch (err) {
                setError('Failed to load result');
            } finally {
                setIsLoading(false);
            }
        };
        if (params.id) fetchResult();
    }, [params.id]);

    const handleDelete = async () => {
        if (!result || !confirm('Delete this result?')) return;
        try {
            await deleteResult(result.id);
            router.push(ROUTES.HISTORY);
        } catch (err) {
            console.error('Delete failed:', err);
        }
    };

    if (isLoading) {
        return (
            <div className="min-h-screen py-12">
                <div className="container-custom max-w-4xl">
                    <Skeleton className="h-8 w-32 mb-8" />
                    <Skeleton className="h-96 rounded-2xl" />
                </div>
            </div>
        );
    }

    if (error || !result) {
        return (
            <div className="min-h-screen py-12 flex items-center justify-center">
                <Card className="p-8 text-center">
                    <p className="text-danger-400 mb-4">{error || 'Not found'}</p>
                    <Link href={ROUTES.HISTORY}>
                        <Button variant="secondary">Back to History</Button>
                    </Link>
                </Card>
            </div>
        );
    }

    return (
        <div className="min-h-screen py-8 md:py-12">
            <div className="container-custom max-w-4xl">
                <Link href={ROUTES.HISTORY}>
                    <Button variant="ghost" size="sm" className="mb-6">
                        <ArrowLeft className="w-4 h-4 mr-2" />
                        Back to History
                    </Button>
                </Link>

                <AnalysisResultCard result={result} showDetails={true} />

                <div className="flex justify-center gap-4 mt-8">
                    <Button variant="secondary" onClick={() => downloadJSON(result, `analysis-${result.id}.json`)}>
                        <Download className="w-4 h-4 mr-2" />
                        Export
                    </Button>
                    <Button variant="ghost" onClick={handleDelete} className="text-danger-400">
                        <Trash2 className="w-4 h-4 mr-2" />
                        Delete
                    </Button>
                </div>
            </div>
        </div>
    );
}
