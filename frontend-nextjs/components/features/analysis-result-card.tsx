'use client';

import { useRef, useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import {
    ShieldCheck,
    ShieldAlert,
    Clock,
    Cpu,
    Database,
    BarChart3,
    Download,
    Share2,
    ChevronRight,
    Film,
    Image as ImageIcon,
    Users,
    FileImage,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils/cn';
import { formatConfidence, formatProcessingTime, formatFileSize, formatRelativeTime } from '@/lib/utils/format';
import { downloadBlob } from '@/lib/utils/download';
import { ROUTES } from '@/lib/constants';
import type { AnalysisResult } from '@/types/analysis';

interface AnalysisResultCardProps {
    result: AnalysisResult;
    showDetails?: boolean;
    className?: string;
}

export function AnalysisResultCard({
    result,
    showDetails = true,
    className,
}: AnalysisResultCardProps) {
    const cardRef = useRef<HTMLDivElement>(null);
    const [exporting, setExporting] = useState(false);
    
    const prediction = result.prediction || (result.confidence > 0.5 ? 'fake' : 'real');
    const isReal = prediction === 'real';
    const confidencePercent = (result.confidence || 0) * 100;

    const handleExportImage = async () => {
        if (!cardRef.current || exporting) return;
        
        setExporting(true);
        try {
            const html2canvas = (await import('html2canvas')).default;
            const canvas = await html2canvas(cardRef.current, {
                backgroundColor: '#0a0e1a',
                scale: 2,
                logging: false,
                useCORS: true,
            });
            
            canvas.toBlob((blob) => {
                if (blob) {
                    downloadBlob(blob, `deepfake-analysis-${result.id}.png`);
                }
            }, 'image/png', 1.0);
        } catch (error) {
            console.error('Export failed:', error);
        } finally {
            setExporting(false);
        }
    };



    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={className}
        >
            <Card
                ref={cardRef}
                variant={isReal ? 'success' : 'danger'}
                className={cn(
                    'overflow-hidden',
                    isReal ? 'shadow-glow-real' : 'shadow-glow-fake'
                )}
            >
                {/* Header with result */}
                <div
                    className={cn(
                        'p-6 text-center relative overflow-hidden',
                        isReal
                            ? 'bg-gradient-to-br from-success-400/10 to-accent-cyan/10'
                            : 'bg-gradient-to-br from-danger-400/10 to-neon-orange/10'
                    )}
                >
                    {/* Decorative orbs */}
                    <div className="absolute inset-0 overflow-hidden pointer-events-none">
                        <div
                            className={cn(
                                'absolute -top-10 -right-10 w-40 h-40 rounded-full blur-3xl',
                                isReal ? 'bg-success-400/20' : 'bg-danger-400/20'
                            )}
                        />
                    </div>

                    <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ type: 'spring', stiffness: 200, damping: 15 }}
                        className="relative"
                    >
                        <div
                            className={cn(
                                'w-20 h-20 mx-auto rounded-2xl flex items-center justify-center mb-4',
                                isReal ? 'bg-success-400/20' : 'bg-danger-400/20',
                                isReal ? 'shadow-glow-real' : 'shadow-glow-fake'
                            )}
                        >
                            {isReal ? (
                                <ShieldCheck className="w-10 h-10 text-success-400" />
                            ) : (
                                <ShieldAlert className="w-10 h-10 text-danger-400" />
                            )}
                        </div>

                        <h2 className="text-3xl font-bold font-display mb-2">
                            <span className={isReal ? 'text-gradient-real' : 'text-gradient-fake'}>
                                {isReal ? 'Likely Real' : 'Likely Fake'}
                            </span>
                        </h2>

                        <div className="flex items-center justify-center gap-2">
                            <span className={cn('text-5xl font-bold font-display', isReal ? 'text-success-400' : 'text-danger-400')}>
                                {confidencePercent.toFixed(1)}%
                            </span>
                            <span className="text-text-muted">confidence</span>
                        </div>

                        <Badge variant={isReal ? 'success' : 'danger'} className="mt-4">
                            {isReal ? 'No Manipulation Detected' : 'Manipulation Detected'}
                        </Badge>
                    </motion.div>
                </div>

                {showDetails && (
                    <>
                        <Separator />

                        <CardContent className="p-6">
                            {/* File Info */}
                            <div className="flex items-start gap-4 mb-6">
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
                                <div className="flex-1 min-w-0">
                                    <p className="font-medium text-text-primary truncate">{result.filename || 'Unknown file'}</p>
                                    <p className="text-sm text-text-muted">
                                        {formatFileSize(result.file_size)} • {(result.file_type || 'image').toUpperCase()}
                                    </p>
                                </div>
                            </div>

                            {/* Stats Grid */}
                            <div className="grid grid-cols-2 gap-4 mb-6">
                                <div className="p-3 rounded-lg bg-dark-hover/50">
                                    <div className="flex items-center gap-2 text-text-muted mb-1">
                                        <Clock className="w-4 h-4" />
                                        <span className="text-xs">Processing Time</span>
                                    </div>
                                    <p className="font-semibold text-text-primary">
                                        {formatProcessingTime(result.processing_time)}
                                    </p>
                                </div>

                                <div className="p-3 rounded-lg bg-dark-hover/50">
                                    <div className="flex items-center gap-2 text-text-muted mb-1">
                                        <Users className="w-4 h-4" />
                                        <span className="text-xs">Faces Detected</span>
                                    </div>
                                    <p className="font-semibold text-text-primary">{result.faces_detected != null ? result.faces_detected : '-'}</p>
                                </div>

                                <div className="p-3 rounded-lg bg-dark-hover/50">
                                    <div className="flex items-center gap-2 text-text-muted mb-1">
                                        <Cpu className="w-4 h-4" />
                                        <span className="text-xs">Model</span>
                                    </div>
                                    <p className="font-semibold text-text-primary text-sm truncate">
                                        {result.model_name || 'Unknown'}
                                    </p>
                                </div>

                                <div className="p-3 rounded-lg bg-dark-hover/50">
                                    <div className="flex items-center gap-2 text-text-muted mb-1">
                                        <Database className="w-4 h-4" />
                                        <span className="text-xs">Dataset</span>
                                    </div>
                                    <p className="font-semibold text-text-primary">{result.dataset || 'DFDC'}</p>
                                </div>
                            </div>

                            {/* Video frame analysis */}
                            {result.frame_predictions && result.frame_predictions.length > 0 && (
                                <div className="mb-6">
                                    <div className="flex items-center gap-2 mb-3">
                                        <BarChart3 className="w-4 h-4 text-accent-cyan" />
                                        <span className="text-sm font-medium text-text-primary">
                                            Frame Analysis ({result.frame_predictions.length} frames)
                                        </span>
                                    </div>
                                    <div className="h-16 flex items-end gap-0.5">
                                        {result.frame_predictions.slice(0, 50).map((frame, idx) => (
                                            <div
                                                key={idx}
                                                className={cn(
                                                    'flex-1 rounded-t transition-all duration-200 hover:opacity-100',
                                                    (frame.prediction || 'real') === 'fake'
                                                        ? 'bg-danger-400'
                                                        : 'bg-success-400',
                                                    'opacity-70'
                                                )}
                                                style={{ height: `${frame.confidence * 100}%` }}
                                                title={`Frame ${frame.frame}: ${formatConfidence(frame.confidence)}`}
                                            />
                                        ))}
                                    </div>
                                    <div className="flex justify-between mt-1 text-xs text-text-muted">
                                        <span>Frame 1</span>
                                        <span>Frame {result.frame_predictions.length}</span>
                                    </div>
                                </div>
                            )}

                            {/* Actions */}
                            <div className="flex items-center gap-3">
                                <Button 
                                    variant="secondary" 
                                    size="sm" 
                                    onClick={handleExportImage}
                                    disabled={exporting}
                                >
                                    <FileImage className="w-4 h-4 mr-2" />
                                    {exporting ? 'Exporting...' : 'Export PNG'}
                                </Button>
                                <Link href={ROUTES.RESULT_DETAIL(result.id)} className="ml-auto">
                                    <Button variant="ghost" size="sm">
                                        View Details
                                        <ChevronRight className="w-4 h-4 ml-1" />
                                    </Button>
                                </Link>
                            </div>

                            {/* Timestamp */}
                            <p className="text-xs text-text-dim mt-4 text-center">
                                Analyzed {formatRelativeTime(result.created_at)}
                            </p>
                        </CardContent>
                    </>
                )}
            </Card>
        </motion.div>
    );
}
