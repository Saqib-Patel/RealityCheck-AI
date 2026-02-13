'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';
import { Shield, Menu, X } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils/cn';
import { ROUTES } from '@/lib/constants';

const navLinks = [
    { href: ROUTES.HOME, label: 'Home' },
    { href: ROUTES.ANALYZE, label: 'Analyze' },
    { href: ROUTES.HOW_IT_WORKS, label: 'How It Works' },
    { href: ROUTES.HISTORY, label: 'History' },
];

export function Header() {
    const pathname = usePathname();
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    return (
        <header className="fixed top-0 left-0 right-0 z-50">
            {/* Glassmorphism background */}
            <div className="absolute inset-0 bg-dark-void/70 backdrop-blur-xl border-b border-white/5" />

            <nav className="container-custom relative">
                <div className="flex items-center justify-between h-16 md:h-20">
                    {/* Logo */}
                    <Link href="/" className="flex items-center gap-2 group">
                        <motion.div
                            whileHover={{ scale: 1.05, rotate: 5 }}
                            className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-cyan to-accent-magenta p-[1px] shadow-glow-cyan"
                        >
                            <div className="w-full h-full rounded-xl bg-dark-void flex items-center justify-center">
                                <Shield className="w-5 h-5 text-accent-cyan" />
                            </div>
                        </motion.div>
                        <span className="font-display font-bold text-lg text-gradient-primary hidden sm:block">
                            DeepFake Hub
                        </span>
                    </Link>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center gap-1">
                        {navLinks.map((link) => {
                            const isActive = pathname === link.href;
                            return (
                                <Link
                                    key={link.href}
                                    href={link.href}
                                    className={cn(
                                        'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                                        isActive
                                            ? 'text-accent-cyan bg-accent-cyan/10'
                                            : 'text-text-secondary hover:text-text-primary hover:bg-dark-hover'
                                    )}
                                >
                                    {link.label}
                                    {isActive && (
                                        <motion.div
                                            layoutId="activeNav"
                                            className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-accent-cyan to-accent-magenta"
                                            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                                        />
                                    )}
                                </Link>
                            );
                        })}
                    </div>

                    {/* CTA Button */}
                    <div className="hidden md:block">
                        <Link href={ROUTES.ANALYZE}>
                            <Button size="sm" className="animate-glow-pulse">
                                <Shield className="w-4 h-4 mr-2" />
                                Start Detection
                            </Button>
                        </Link>
                    </div>

                    {/* Mobile Menu Button */}
                    <button
                        className="md:hidden p-2 rounded-lg text-text-secondary hover:text-text-primary hover:bg-dark-hover transition-colors"
                        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                    >
                        {isMobileMenuOpen ? (
                            <X className="w-6 h-6" />
                        ) : (
                            <Menu className="w-6 h-6" />
                        )}
                    </button>
                </div>

                {/* Mobile Menu */}
                {isMobileMenuOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="md:hidden absolute top-full left-0 right-0 bg-dark-card/95 backdrop-blur-xl border-b border-dark-border p-4"
                    >
                        <div className="flex flex-col gap-2">
                            {navLinks.map((link) => {
                                const isActive = pathname === link.href;
                                return (
                                    <Link
                                        key={link.href}
                                        href={link.href}
                                        onClick={() => setIsMobileMenuOpen(false)}
                                        className={cn(
                                            'px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200',
                                            isActive
                                                ? 'text-accent-cyan bg-accent-cyan/10'
                                                : 'text-text-secondary hover:text-text-primary hover:bg-dark-hover'
                                        )}
                                    >
                                        {link.label}
                                    </Link>
                                );
                            })}
                            <div className="pt-2 mt-2 border-t border-dark-border">
                                <Link href={ROUTES.ANALYZE} onClick={() => setIsMobileMenuOpen(false)}>
                                    <Button className="w-full">
                                        <Shield className="w-4 h-4 mr-2" />
                                        Start Detection
                                    </Button>
                                </Link>
                            </div>
                        </div>
                    </motion.div>
                )}
            </nav>
        </header>
    );
}
