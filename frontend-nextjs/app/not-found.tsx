'use client';

import Link from 'next/link';
import { Home, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ROUTES } from '@/lib/constants';

export default function NotFound() {
    return (
        <div className="min-h-screen flex items-center justify-center py-12">
            <div className="text-center px-4">
                <div className="mb-8">
                    <span className="text-8xl font-bold font-display text-gradient-aurora">404</span>
                </div>
                <h1 className="text-2xl md:text-3xl font-bold text-text-primary mb-4 font-display">
                    Page Not Found
                </h1>
                <p className="text-text-secondary mb-8 max-w-md mx-auto">
                    The page you&apos;re looking for doesn&apos;t exist or has been moved.
                </p>
                <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                    <Link href={ROUTES.HOME}>
                        <Button>
                            <Home className="w-4 h-4 mr-2" />
                            Go Home
                        </Button>
                    </Link>
                    <Button variant="secondary" onClick={() => window.history.back()}>
                        <ArrowLeft className="w-4 h-4 mr-2" />
                        Go Back
                    </Button>
                </div>
            </div>
        </div>
    );
}
