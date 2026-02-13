import Link from 'next/link';
import { Shield, Github, Twitter, Linkedin, Heart } from 'lucide-react';
import { ROUTES, SOCIAL_LINKS, APP_NAME } from '@/lib/constants';

const footerLinks = {
    navigation: [
        { href: ROUTES.HOME, label: 'Home' },
        { href: ROUTES.ANALYZE, label: 'Analyze' },
        { href: ROUTES.HOW_IT_WORKS, label: 'How It Works' },
        { href: ROUTES.HISTORY, label: 'History' },
    ],
    resources: [
        { href: SOCIAL_LINKS.github, label: 'GitHub Profile', external: true },
        { href: 'https://github.com/Saqib-Patel/DeepFake-Detection-Hub', label: 'Project Repository', external: true },
    ],
};

const socialLinks = [
    { href: SOCIAL_LINKS.github, icon: Github, label: 'GitHub' },
    { href: SOCIAL_LINKS.twitter, icon: Twitter, label: 'Twitter' },
    { href: SOCIAL_LINKS.linkedin, icon: Linkedin, label: 'LinkedIn' },
];

export function Footer() {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="relative border-t border-dark-border bg-dark-deep/50">
            {/* Gradient decoration */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-accent-cyan/50 to-transparent" />

            <div className="container-custom py-12 md:py-16">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12">
                    {/* Brand */}
                    <div className="col-span-1">
                        <Link href="/" className="flex items-center gap-2 mb-4">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-cyan to-accent-magenta p-[1px]">
                                <div className="w-full h-full rounded-xl bg-dark-void flex items-center justify-center">
                                    <Shield className="w-5 h-5 text-accent-cyan" />
                                </div>
                            </div>
                            <span className="font-display font-bold text-lg text-gradient-primary">
                                DeepFake Hub
                            </span>
                        </Link>
                        <p className="text-sm text-text-secondary mb-4 max-w-xs">
                            AI-powered deepfake detection using state-of-the-art deep learning models.
                        </p>
                        {/* Social Links */}
                        <div className="flex items-center gap-3">
                            {socialLinks.map((social) => (
                                <a
                                    key={social.label}
                                    href={social.href}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="w-9 h-9 rounded-lg bg-dark-elevated border border-dark-border flex items-center justify-center text-text-muted hover:text-accent-cyan hover:border-accent-cyan/50 transition-all duration-200"
                                    aria-label={social.label}
                                >
                                    <social.icon className="w-4 h-4" />
                                </a>
                            ))}
                        </div>
                    </div>

                    {/* Navigation */}
                    <div>
                        <h4 className="font-semibold text-text-primary mb-4 font-display">Navigation</h4>
                        <ul className="space-y-2">
                            {footerLinks.navigation.map((link) => (
                                <li key={link.href}>
                                    <Link
                                        href={link.href}
                                        className="text-sm text-text-secondary hover:text-accent-cyan transition-colors duration-200"
                                    >
                                        {link.label}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Resources */}
                    <div>
                        <h4 className="font-semibold text-text-primary mb-4 font-display">Resources</h4>
                        <ul className="space-y-2">
                            {footerLinks.resources.map((link) => (
                                <li key={link.label}>
                                    <a
                                        href={link.href}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-sm text-text-secondary hover:text-accent-cyan transition-colors duration-200 flex items-center gap-1"
                                    >
                                        {link.label}
                                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                        </svg>
                                    </a>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                {/* Bottom Bar */}
                <div className="mt-12 pt-8 border-t border-dark-border flex flex-col md:flex-row items-center justify-between gap-4">
                    <p className="text-sm text-text-muted text-center md:text-left">
                        © {currentYear} Mohammed Saqib Patel. All rights reserved.
                    </p>
                    <p className="text-sm text-text-muted flex items-center gap-1">
                        Built by Mohammed Saqib Patel
                    </p>
                </div>
            </div>
        </footer>
    );
}
