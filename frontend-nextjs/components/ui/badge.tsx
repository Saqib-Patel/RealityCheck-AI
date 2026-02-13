import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils/cn';

const badgeVariants = cva(
    'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-all duration-200',
    {
        variants: {
            variant: {
                default:
                    'bg-dark-elevated border border-accent-cyan/30 text-accent-cyan',
                cyan:
                    'bg-accent-cyan/10 border border-accent-cyan/30 text-accent-cyan',
                magenta:
                    'bg-accent-magenta/10 border border-accent-magenta/30 text-accent-magenta',
                purple:
                    'bg-accent-purple/10 border border-accent-purple/30 text-accent-purple',
                success:
                    'bg-success-400/10 border border-success-400/30 text-success-400',
                danger:
                    'bg-danger-400/10 border border-danger-400/30 text-danger-400',
                warning:
                    'bg-neon-yellow/10 border border-neon-yellow/30 text-neon-yellow',
                outline:
                    'border border-dark-border text-text-secondary',
                solid:
                    'bg-gradient-to-r from-accent-cyan to-accent-magenta text-dark-void font-semibold',
                real:
                    'badge-real',
                fake:
                    'badge-fake',
            },
        },
        defaultVariants: {
            variant: 'default',
        },
    }
);

export interface BadgeProps
    extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> { }

function Badge({ className, variant, ...props }: BadgeProps) {
    return (
        <div className={cn(badgeVariants({ variant }), className)} {...props} />
    );
}

export { Badge, badgeVariants };
