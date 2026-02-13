'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import {
    History,
    Search,
    Filter,
    Trash2,
    ShieldCheck,
    ShieldAlert,
    Clock,
    Image as ImageIcon,
    Film,
    ArrowRight,
} from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { useResults } from '@/hooks/use-results';
import { useDebounce } from '@/hooks/use-debounce';
import { cn } from '@/lib/utils/cn';
import { formatRelativeTime, formatConfidence } from '@/lib/utils/format';
import { ROUTES } from '@/lib/constants';
import type { AnalysisResult } from '@/types/analysis';

export default function HistoryPage() {
    const { results, isLoading, error, deleteResult } = useResults();
    const [searchQuery, setSearchQuery] = useState('');
    const [filterPrediction, setFilterPrediction] = useState<string>('all');
    const debouncedSearch = useDebounce(searchQuery, 300);

    const filteredResults = results.filter((result) => {
        // Search filter
        if (debouncedSearch && !(result.filename || '').toLowerCase().includes(debouncedSearch.toLowerCase())) {
            return false;
        }
        // Prediction filter
        if (filterPrediction !== 'all' && result.prediction !== filterPrediction) {
            return false;
        }
        return true;
    });

    const handleDelete = async (id: string) => {
        if (confirm('Are you sure you want to delete this result?')) {
            try {
                await deleteResult(id);
            } catch (err) {
                console.error('Failed to delete:', err);
            }
        }
    };

    return (
        <div className="min-h-screen py-8 md:py-12">
            <div className="container-custom max-w-5xl">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8"
                >
                    <div className="flex items-center gap-3 mb-2">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-purple/20 to-accent-magenta/20 flex items-center justify-center border border-accent-purple/30">
                            <History className="w-6 h-6 text-accent-purple" />
                        </div>
                        <div>
                            <h1 className="text-2xl md:text-3xl font-bold text-text-primary font-display">
                                Analysis History
                            </h1>
                            <p className="text-text-secondary text-sm">
                                View and manage your previous analysis results
                            </p>
                        </div>
                    </div>
                </motion.div>

                {/* Filters */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="flex flex-col sm:flex-row gap-4 mb-6"
                >
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-muted" />
                        <Input
                            placeholder="Search by filename..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="pl-10"
                        />
                    </div>
                    <Select value={filterPrediction} onValueChange={setFilterPrediction}>
                        <SelectTrigger className="w-full sm:w-40">
                            <Filter className="w-4 h-4 mr-2" />
                            <SelectValue placeholder="Filter" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="all">All Results</SelectItem>
                            <SelectItem value="real">Real</SelectItem>
                            <SelectItem value="fake">Fake</SelectItem>
                        </SelectContent>
                    </Select>
                </motion.div>

                {/* Results count */}
                <div className="text-sm text-text-muted mb-4">
                    {filteredResults.length} {filteredResults.length === 1 ? 'result' : 'results'}
                    {debouncedSearch && ` for "${debouncedSearch}"`}
                </div>

                {/* Loading state */}
                {isLoading && (
                    <div className="space-y-4">
                        {[1, 2, 3].map((i) => (
                            <Card key={i} className="p-4">
                                <div className="flex items-center gap-4">
                                    <Skeleton className="w-12 h-12 rounded-xl" />
                                    <div className="flex-1 space-y-2">
                                        <Skeleton className="h-4 w-1/3" />
                                        <Skeleton className="h-3 w-1/4" />
                                    </div>
                                    <Skeleton className="h-8 w-20" />
                                </div>
                            </Card>
                        ))}
                    </div>
                )}

                {/* Error state */}
                {error && (
                    <Card variant="danger" className="p-8 text-center">
                        <p className="text-danger-400 mb-4">{error}</p>
                        <Button variant="secondary" onClick={() => window.location.reload()}>
                            Try Again
                        </Button>
                    </Card>
                )}

                {/* Empty state */}
                {!isLoading && !error && filteredResults.length === 0 && (
                    <Card variant="glass" className="p-12 text-center">
                        <History className="w-16 h-16 mx-auto text-text-muted mb-4" />
                        <h3 className="text-lg font-semibold text-text-primary mb-2">
                            No results found
                        </h3>
                        <p className="text-text-secondary mb-6">
                            {debouncedSearch
                                ? 'Try adjusting your search or filters'
                                : "You haven't analyzed any files yet"}
                        </p>
                        <Link href={ROUTES.ANALYZE}>
                            <Button>
                                Analyze Your First File
                                <ArrowRight className="w-4 h-4 ml-2" />
                            </Button>
                        </Link>
                    </Card>
                )}

                {/* Results list */}
                {!isLoading && !error && filteredResults.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                        className="space-y-4"
                    >
                        {filteredResults.map((result, index) => (
                            <ResultRow
                                key={result.id}
                                result={result}
                                index={index}
                                onDelete={() => handleDelete(result.id)}
                            />
                        ))}
                    </motion.div>
                )}
            </div>
        </div>
    );
}

interface ResultRowProps {
    result: AnalysisResult;
    index: number;
    onDelete: () => void;
}

function ResultRow({ result, index, onDelete }: ResultRowProps) {
    const isReal = (result.prediction || 'real') === 'real';

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
        >
            <Card hover className="p-4">
                <div className="flex items-center gap-4">
                    {/* File type icon */}
                    <div
                        className={cn(
                            'w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0',
                            result.file_type === 'video' ? 'bg-accent-magenta/10' : 'bg-accent-cyan/10'
                        )}
                    >
                        {result.file_type === 'video' ? (
                            <Film className="w-6 h-6 text-accent-magenta" />
                        ) : (
                            <ImageIcon className="w-6 h-6 text-accent-cyan" />
                        )}
                    </div>

                    {/* File info */}
                    <div className="flex-1 min-w-0">
                        <p className="font-medium text-text-primary truncate">{result.filename || 'Unknown file'}</p>
                        <div className="flex items-center gap-3 text-sm text-text-muted">
                            <span className="flex items-center gap-1">
                                <Clock className="w-3.5 h-3.5" />
                                {formatRelativeTime(result.created_at)}
                            </span>
                            <span>{result.model_name || 'Unknown Model'}</span>
                        </div>
                    </div>

                    {/* Prediction badge */}
                    <Badge variant={isReal ? 'success' : 'danger'} className="flex items-center gap-1.5">
                        {isReal ? (
                            <ShieldCheck className="w-4 h-4" />
                        ) : (
                            <ShieldAlert className="w-4 h-4" />
                        )}
                        {formatConfidence(result.confidence)}
                    </Badge>

                    {/* Actions */}
                    <div className="flex items-center gap-2">
                        <Link href={ROUTES.RESULT_DETAIL(result.id)}>
                            <Button variant="ghost" size="sm">
                                View
                                <ArrowRight className="w-4 h-4 ml-1" />
                            </Button>
                        </Link>
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={onDelete}
                            className="text-text-muted hover:text-danger-400"
                        >
                            <Trash2 className="w-4 h-4" />
                        </Button>
                    </div>
                </div>
            </Card>
        </motion.div>
    );
}
