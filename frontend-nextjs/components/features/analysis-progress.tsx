'use client';

import { motion } from 'framer-motion';
import { Loader2, CheckCircle2, Brain, Scan, BarChart3 } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils/cn';

interface AnalysisProgressProps {
    progress: number;
    status: 'uploading' | 'analyzing' | 'processing' | 'complete';
    currentStep?: string;
    eta?: number;
    className?: string;
}

const steps = [
    { id: 'upload', label: 'Uploading', icon: Loader2 },
    { id: 'detect', label: 'Detecting Faces', icon: Scan },
    { id: 'analyze', label: 'AI Analysis', icon: Brain },
    { id: 'results', label: 'Generating Results', icon: BarChart3 },
];

export function AnalysisProgress({
    progress,
    status,
    currentStep = 'Analyzing...',
    eta,
    className,
}: AnalysisProgressProps) {
    const getStepStatus = (stepIndex: number) => {
        if (status === 'complete') return 'complete';
        if (status === 'uploading' && stepIndex === 0) return 'active';
        if (status === 'analyzing') {
            if (progress < 30 && stepIndex === 1) return 'active';
            if (progress >= 30 && progress < 70 && stepIndex === 2) return 'active';
            if (progress >= 70 && stepIndex === 3) return 'active';
            if (progress >= 30 && stepIndex < 2) return 'complete';
            if (progress >= 70 && stepIndex < 3) return 'complete';
        }
        return 'pending';
    };

    return (
        <Card variant="glass" className={cn('overflow-hidden', className)}>
            <CardContent className="p-6">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                        <motion.div
                            animate={{ rotate: status !== 'complete' ? 360 : 0 }}
                            transition={{ duration: 1, repeat: status !== 'complete' ? Infinity : 0, ease: 'linear' }}
                            className={cn(
                                'w-12 h-12 rounded-xl flex items-center justify-center',
                                status === 'complete'
                                    ? 'bg-success-400/20'
                                    : 'bg-gradient-to-br from-accent-cyan/20 to-accent-magenta/20'
                            )}
                        >
                            {status === 'complete' ? (
                                <CheckCircle2 className="w-6 h-6 text-success-400" />
                            ) : (
                                <Brain className="w-6 h-6 text-accent-cyan" />
                            )}
                        </motion.div>
                        <div>
                            <h3 className="text-lg font-semibold text-text-primary font-display">
                                {status === 'complete' ? 'Analysis Complete!' : currentStep}
                            </h3>
                            {eta && status !== 'complete' && (
                                <p className="text-sm text-text-muted">
                                    Estimated time: {Math.ceil(eta)}s remaining
                                </p>
                            )}
                        </div>
                    </div>
                    <div className="text-right">
                        <span className="text-2xl font-bold text-gradient-aurora font-display">
                            {Math.round(progress)}%
                        </span>
                    </div>
                </div>

                {/* Progress bar */}
                <div className="mb-6">
                    <Progress value={progress} variant="aurora" className="h-3" />
                </div>

                {/* Steps */}
                <div className="grid grid-cols-4 gap-2">
                    {steps.map((step, index) => {
                        const stepStatus = getStepStatus(index);
                        const Icon = step.icon;

                        return (
                            <motion.div
                                key={step.id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className={cn(
                                    'flex flex-col items-center p-3 rounded-xl transition-all duration-300',
                                    stepStatus === 'active' && 'bg-accent-cyan/10',
                                    stepStatus === 'complete' && 'bg-success-400/10'
                                )}
                            >
                                <div
                                    className={cn(
                                        'w-10 h-10 rounded-lg flex items-center justify-center mb-2 transition-all duration-300',
                                        stepStatus === 'pending' && 'bg-dark-hover text-text-muted',
                                        stepStatus === 'active' && 'bg-accent-cyan/20 text-accent-cyan',
                                        stepStatus === 'complete' && 'bg-success-400/20 text-success-400'
                                    )}
                                >
                                    {stepStatus === 'complete' ? (
                                        <CheckCircle2 className="w-5 h-5" />
                                    ) : stepStatus === 'active' ? (
                                        <motion.div
                                            animate={{ rotate: 360 }}
                                            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                                        >
                                            <Icon className="w-5 h-5" />
                                        </motion.div>
                                    ) : (
                                        <Icon className="w-5 h-5" />
                                    )}
                                </div>
                                <span
                                    className={cn(
                                        'text-xs font-medium text-center transition-colors duration-300',
                                        stepStatus === 'pending' && 'text-text-muted',
                                        stepStatus === 'active' && 'text-accent-cyan',
                                        stepStatus === 'complete' && 'text-success-400'
                                    )}
                                >
                                    {step.label}
                                </span>
                            </motion.div>
                        );
                    })}
                </div>

                {/* Animated dots */}
                {status !== 'complete' && (
                    <div className="flex items-center justify-center gap-1 mt-6">
                        {[0, 1, 2].map((i) => (
                            <motion.div
                                key={i}
                                animate={{ opacity: [0.3, 1, 0.3] }}
                                transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2 }}
                                className="w-2 h-2 rounded-full bg-accent-cyan"
                            />
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
