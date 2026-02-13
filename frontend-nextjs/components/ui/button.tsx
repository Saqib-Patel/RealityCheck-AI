import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils/cn';

const buttonVariants = cva(
    'inline-flex items-center justify-center whitespace-nowrap rounded-xl text-sm font-semibold transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-cyan focus-visible:ring-offset-2 focus-visible:ring-offset-dark-void disabled:pointer-events-none disabled:opacity-50',
    {
        variants: {
            variant: {
                default:
                    'bg-gradient-to-r from-accent-cyan to-accent-magenta text-dark-void hover:shadow-glow-cyan hover:-translate-y-0.5',
                secondary:
                    'bg-dark-elevated border border-accent-cyan/30 text-text-primary hover:border-accent-cyan hover:shadow-glow-cyan hover:-translate-y-0.5',
                outline:
                    'border border-dark-border bg-transparent text-text-primary hover:bg-dark-elevated hover:border-accent-cyan/50',
                ghost:
                    'text-text-secondary hover:text-text-primary hover:bg-dark-elevated',
                destructive:
                    'bg-gradient-to-r from-danger-400 to-neon-orange text-white hover:shadow-glow-fake hover:-translate-y-0.5',
                success:
                    'bg-gradient-to-r from-success-400 to-accent-cyan text-dark-void hover:shadow-glow-real hover:-translate-y-0.5',
                link:
                    'text-accent-cyan underline-offset-4 hover:underline hover:text-accent-cyan-dim',
            },
            size: {
                default: 'h-10 px-4 py-2',
                sm: 'h-9 px-3 text-xs',
                lg: 'h-12 px-6 text-base',
                xl: 'h-14 px-8 text-lg',
                icon: 'h-10 w-10',
            },
        },
        defaultVariants: {
            variant: 'default',
            size: 'default',
        },
    }
);

export interface ButtonProps
    extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
    asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant, size, asChild = false, ...props }, ref) => {
        const Comp = asChild ? Slot : 'button';
        return (
            <Comp
                className={cn(buttonVariants({ variant, size, className }))}
                ref={ref}
                {...props}
            />
        );
    }
);
Button.displayName = 'Button';

export { Button, buttonVariants };
