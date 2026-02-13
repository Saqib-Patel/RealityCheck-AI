'use client';

import { useState, useEffect } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Zap } from 'lucide-react';
import apiClient from '@/lib/api/client';

/**
 * ServerStatus component - Shows backend status and handles Render free tier cold starts
 * Displays a friendly message when the backend is waking up from sleep
 */
export function ServerStatus() {
    const [status, setStatus] = useState<'checking' | 'online' | 'waking' | 'offline'>('checking');
    const [wakingTime, setWakingTime] = useState(0);

    useEffect(() => {
        let interval: NodeJS.Timeout;
        let mounted = true;

        const checkHealth = async () => {
            const startTime = Date.now();
            if (mounted) setStatus('checking');

            try {
                await apiClient.get('/health/', { timeout: 15000 });
                if (mounted) setStatus('online');
            } catch (error) {
                if (!mounted) return;
                const elapsed = Date.now() - startTime;

                // If timeout or network error after 10s, likely cold start
                if (elapsed > 10000) {
                    setStatus('waking');
                    setWakingTime(Math.floor(elapsed / 1000));
                } else {
                    setStatus('offline');
                }
            }
        };

        // Check once on mount
        checkHealth();

        // Re-check every 60 seconds (not 30) to reduce backend load on free tier
        interval = setInterval(() => {
            checkHealth();
        }, 60000);

        return () => {
            mounted = false;
            clearInterval(interval);
        };
    }, []); // stable deps – no re-run on status change

    // Don't show anything if online or checking initially
    if (status === 'online' || (status === 'checking' && wakingTime === 0)) {
        return null;
    }

    return (
        <div className="fixed top-20 left-1/2 -translate-x-1/2 z-50 w-full max-w-md px-4">
            {status === 'waking' && (
                <Alert className="bg-accent-cyan/10 border-accent-cyan/30 backdrop-blur-sm">
                    <Loader2 className="h-4 w-4 animate-spin text-accent-cyan" />
                    <AlertDescription className="text-text-primary">
                        <span className="font-semibold">Server is waking up...</span>
                        <p className="text-sm text-text-secondary mt-1">
                            Free tier backend is starting (this takes 30-60 seconds on first request).
                            Thanks for your patience! ⏳
                        </p>
                    </AlertDescription>
                </Alert>
            )}

            {status === 'offline' && (
                <Alert variant="destructive" className="backdrop-blur-sm">
                    <Zap className="h-4 w-4" />
                    <AlertDescription>
                        <span className="font-semibold">Backend temporarily unavailable</span>
                        <p className="text-sm mt-1">
                            Retrying connection... If this persists, the backend may be down.
                        </p>
                    </AlertDescription>
                </Alert>
            )}
        </div>
    );
}
