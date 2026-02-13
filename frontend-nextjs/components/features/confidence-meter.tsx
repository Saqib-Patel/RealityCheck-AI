'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils/cn';

interface ConfidenceMeterProps {
    confidence: number;
    prediction: 'real' | 'fake';
    size?: 'sm' | 'md' | 'lg';
    showLabel?: boolean;
    className?: string;
}

export function ConfidenceMeter({
    confidence,
    prediction,
    size = 'md',
    showLabel = true,
    className,
}: ConfidenceMeterProps) {
    const isReal = prediction === 'real';
    const percentage = confidence * 100;

    const sizeConfig = {
        sm: { diameter: 80, strokeWidth: 6, fontSize: 'text-lg' },
        md: { diameter: 120, strokeWidth: 8, fontSize: 'text-2xl' },
        lg: { diameter: 160, strokeWidth: 10, fontSize: 'text-4xl' },
    };

    const { diameter, strokeWidth, fontSize } = sizeConfig[size];
    const radius = (diameter - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (percentage / 100) * circumference;

    return (
        <div className={cn('flex flex-col items-center', className)}>
            <div className="relative" style={{ width: diameter, height: diameter }}>
                {/* Background circle */}
                <svg
                    className="transform -rotate-90"
                    width={diameter}
                    height={diameter}
                >
                    <circle
                        cx={diameter / 2}
                        cy={diameter / 2}
                        r={radius}
                        fill="none"
                        stroke="currentColor"
                        strokeWidth={strokeWidth}
                        className="text-dark-hover"
                    />
                    {/* Animated progress circle */}
                    <motion.circle
                        cx={diameter / 2}
                        cy={diameter / 2}
                        r={radius}
                        fill="none"
                        stroke="url(#gradient)"
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        initial={{ strokeDashoffset: circumference }}
                        animate={{ strokeDashoffset: offset }}
                        transition={{ duration: 1.5, ease: 'easeOut' }}
                        style={{
                            strokeDasharray: circumference,
                        }}
                    />
                    {/* Gradient definition */}
                    <defs>
                        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            {isReal ? (
                                <>
                                    <stop offset="0%" stopColor="#00ffc8" />
                                    <stop offset="100%" stopColor="#00f0ff" />
                                </>
                            ) : (
                                <>
                                    <stop offset="0%" stopColor="#ff0066" />
                                    <stop offset="100%" stopColor="#ff6b00" />
                                </>
                            )}
                        </linearGradient>
                    </defs>
                </svg>

                {/* Center content */}
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <motion.span
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.5, type: 'spring' }}
                        className={cn(
                            'font-bold font-display',
                            fontSize,
                            isReal ? 'text-success-400' : 'text-danger-400'
                        )}
                    >
                        {percentage.toFixed(0)}%
                    </motion.span>
                </div>

                {/* Glow effect */}
                <div
                    className={cn(
                        'absolute inset-0 rounded-full blur-xl opacity-30',
                        isReal ? 'bg-success-400' : 'bg-danger-400'
                    )}
                    style={{ transform: 'scale(0.8)' }}
                />
            </div>

            {showLabel && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.8 }}
                    className="mt-3 text-center"
                >
                    <span
                        className={cn(
                            'text-sm font-medium',
                            isReal ? 'text-success-400' : 'text-danger-400'
                        )}
                    >
                        {isReal ? 'Likely Real' : 'Likely Fake'}
                    </span>
                </motion.div>
            )}
        </div>
    );
}
