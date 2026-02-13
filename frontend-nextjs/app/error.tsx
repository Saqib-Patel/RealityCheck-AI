'use client';

import { useEffect } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function Error({
    error,
    reset,
}: {
    error: Error & { digest?: string };
    reset: () => void;
}) {
    useEffect(() => {
        console.error('Application error:', error);
    }, [error]);

    return (
        <div className="min-h-screen flex items-center justify-center py-12">
            <div className="text-center px-4 max-w-md">
                <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-danger-400/10 flex items-center justify-center">
                    <AlertTriangle className="w-10 h-10 text-danger-400" />
                </div>
                <h1 className="text-2xl font-bold text-text-primary mb-4 font-display">
                    Something went wrong
                </h1>
                <p className="text-text-secondary mb-8">
                    An unexpected error occurred. Please try again.
                </p>
                <Button onClick={reset}>
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Try Again
                </Button>
            </div>
        </div>
    );
}
