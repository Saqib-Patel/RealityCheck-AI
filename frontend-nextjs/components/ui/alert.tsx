import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { AlertCircle, CheckCircle2, Info, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils/cn';

const alertVariants = cva(
    'relative w-full rounded-xl border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4',
    {
        variants: {
            variant: {
                default: 'bg-dark-elevated border-dark-border text-text-primary',
                info: 'border-accent-cyan/30 bg-accent-cyan/5 text-accent-cyan [&>svg]:text-accent-cyan',
                success: 'border-success-400/30 bg-success-400/5 text-success-400 [&>svg]:text-success-400',
                warning: 'border-neon-yellow/30 bg-neon-yellow/5 text-neon-yellow [&>svg]:text-neon-yellow',
                destructive: 'border-danger-400/30 bg-danger-400/5 text-danger-400 [&>svg]:text-danger-400',
            },
        },
        defaultVariants: {
            variant: 'default',
        },
    }
);

const Alert = React.forwardRef<
    HTMLDivElement,
    React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>
>(({ className, variant, children, ...props }, ref) => {
    const Icon = {
        default: Info,
        info: Info,
        success: CheckCircle2,
        warning: AlertCircle,
        destructive: XCircle,
    }[variant || 'default'];

    return (
        <div
            ref={ref}
            role="alert"
            className={cn(alertVariants({ variant }), className)}
            {...props}
        >
            <Icon className="h-5 w-5" />
            {children}
        </div>
    );
});
Alert.displayName = 'Alert';

const AlertTitle = React.forwardRef<
    HTMLParagraphElement,
    React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
    <h5
        ref={ref}
        className={cn('mb-1 font-medium leading-none tracking-tight', className)}
        {...props}
    />
));
AlertTitle.displayName = 'AlertTitle';

const AlertDescription = React.forwardRef<
    HTMLParagraphElement,
    React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn('text-sm opacity-90', className)}
        {...props}
    />
));
AlertDescription.displayName = 'AlertDescription';

export { Alert, AlertTitle, AlertDescription };
