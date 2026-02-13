'use client';

import * as React from 'react';
import * as ProgressPrimitive from '@radix-ui/react-progress';
import { cn } from '@/lib/utils/cn';

interface ProgressProps
    extends React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root> {
    variant?: 'default' | 'aurora' | 'success' | 'danger';
    showValue?: boolean;
}

const Progress = React.forwardRef<
    React.ElementRef<typeof ProgressPrimitive.Root>,
    ProgressProps
>(({ className, value, variant = 'default', showValue = false, ...props }, ref) => {
    const getIndicatorClass = () => {
        switch (variant) {
            case 'aurora':
                return 'bg-gradient-to-r from-accent-cyan via-accent-purple to-accent-magenta bg-[length:200%_100%] animate-[progress-shimmer_2s_ease_infinite]';
            case 'success':
                return 'bg-gradient-to-r from-success-400 to-accent-cyan';
            case 'danger':
                return 'bg-gradient-to-r from-danger-400 to-neon-orange';
            default:
                return 'bg-gradient-to-r from-accent-cyan to-accent-magenta';
        }
    };

    return (
        <div className="relative">
            <ProgressPrimitive.Root
                ref={ref}
                className={cn(
                    'relative h-2 w-full overflow-hidden rounded-full bg-dark-elevated',
                    className
                )}
                {...props}
            >
                <ProgressPrimitive.Indicator
                    className={cn(
                        'h-full w-full flex-1 transition-all duration-500 ease-out rounded-full',
                        getIndicatorClass()
                    )}
                    style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
                />
            </ProgressPrimitive.Root>
            {showValue && (
                <span className="absolute right-0 -top-6 text-xs text-text-secondary">
                    {value}%
                </span>
            )}
        </div>
    );
});
Progress.displayName = ProgressPrimitive.Root.displayName;

export { Progress };
