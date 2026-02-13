import * as React from 'react';
import { cn } from '@/lib/utils/cn';

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

const Input = React.forwardRef<HTMLInputElement, InputProps>(
    ({ className, type, ...props }, ref) => {
        return (
            <input
                type={type}
                className={cn(
                    'flex h-11 w-full rounded-xl border border-dark-border bg-dark-elevated px-4 py-2 text-sm text-text-primary placeholder:text-text-muted transition-all duration-200',
                    'focus:outline-none focus:ring-2 focus:ring-accent-cyan/50 focus:border-accent-cyan',
                    'hover:border-accent-cyan/50',
                    'disabled:cursor-not-allowed disabled:opacity-50',
                    'file:border-0 file:bg-transparent file:text-sm file:font-medium',
                    className
                )}
                ref={ref}
                {...props}
            />
        );
    }
);
Input.displayName = 'Input';

export { Input };
