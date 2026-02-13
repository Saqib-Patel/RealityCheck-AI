import type { Variants, Transition } from 'framer-motion';

export const fadeIn: Variants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.6, ease: 'easeOut' } },
};

export const fadeInUp: Variants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: 'easeOut' } },
};

export const scaleIn: Variants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { opacity: 1, scale: 1, transition: { duration: 0.4, ease: 'easeOut' } },
};

export const staggerContainer: Variants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1, delayChildren: 0.2 } },
};

export const staggerItem: Variants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
};

export const slideInFromLeft: Variants = {
    hidden: { opacity: 0, x: -50 },
    visible: { opacity: 1, x: 0, transition: { duration: 0.5, ease: 'easeOut' } },
};

export const slideInFromRight: Variants = {
    hidden: { opacity: 0, x: 50 },
    visible: { opacity: 1, x: 0, transition: { duration: 0.5, ease: 'easeOut' } },
};

export const pop: Variants = {
    initial: { scale: 1 },
    hover: { scale: 1.05 },
    tap: { scale: 0.95 },
};

export const pulseGlow: Variants = {
    initial: {
        boxShadow: '0 0 20px rgba(0, 240, 255, 0.3), 0 0 40px rgba(255, 0, 212, 0.2)',
    },
    animate: {
        boxShadow: [
            '0 0 20px rgba(0, 240, 255, 0.3), 0 0 40px rgba(255, 0, 212, 0.2)',
            '0 0 30px rgba(0, 240, 255, 0.5), 0 0 60px rgba(255, 0, 212, 0.3)',
            '0 0 20px rgba(0, 240, 255, 0.3), 0 0 40px rgba(255, 0, 212, 0.2)',
        ],
        transition: { duration: 2, repeat: Infinity, ease: 'easeInOut' },
    },
};

export const springTransition: Transition = { type: 'spring', stiffness: 300, damping: 20 };
export const smoothTransition: Transition = { type: 'tween', duration: 0.3, ease: 'easeInOut' };

export const pageTransition: Variants = {
    initial: { opacity: 0, y: 10 },
    enter: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } },
    exit: { opacity: 0, y: -10, transition: { duration: 0.3, ease: 'easeIn' } },
};
