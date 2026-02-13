import { Loader2 } from 'lucide-react';

export default function Loading() {
    return (
        <div className="min-h-screen flex items-center justify-center">
            <div className="text-center">
                <Loader2 className="w-12 h-12 text-accent-cyan animate-spin mx-auto mb-4" />
                <p className="text-text-secondary">Loading...</p>
            </div>
        </div>
    );
}
