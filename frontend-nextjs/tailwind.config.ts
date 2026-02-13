import type { Config } from 'tailwindcss';

const config: Config = {
    content: [
        './pages/**/*.{js,ts,jsx,tsx,mdx}',
        './components/**/*.{js,ts,jsx,tsx,mdx}',
        './app/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            colors: {
                'accent-cyan': '#00f0ff',
                'accent-cyan-dim': '#00c4d4',
                'accent-magenta': '#ff00d4',
                'accent-magenta-dim': '#cc00aa',
                'accent-purple': '#a855f7',
                'accent-pink': '#f472b6',
                'neon-lime': '#39ff14',
                'neon-yellow': '#f0ff00',
                'neon-orange': '#ff6b00',
                'neon-blue': '#0066ff',
                'success-400': '#00ffc8',
                'success-500': '#00e6b4',
                'danger-400': '#ff0066',
                'danger-500': '#e6005c',
                'dark-void': '#030308',
                'dark-deep': '#0a0a14',
                'dark-primary': '#0f0f1a',
                'dark-secondary': '#111118',
                'dark-elevated': '#161625',
                'dark-card': '#1a1a2e',
                'dark-hover': '#222238',
                'dark-border': '#2a2a40',
                'text-white': '#ffffff',
                'text-primary': '#f0f0ff',
                'text-secondary': '#a0a0cc',
                'text-muted': '#6a6a8f',
                'text-dim': '#4a4a6a',
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                display: ['Space Grotesk', 'Inter', 'sans-serif'],
            },
            boxShadow: {
                'glow-cyan': '0 0 30px rgba(0, 240, 255, 0.3), 0 0 60px rgba(0, 240, 255, 0.15)',
                'glow-magenta': '0 0 30px rgba(255, 0, 212, 0.3), 0 0 60px rgba(255, 0, 212, 0.15)',
                'glow-real': '0 0 30px rgba(0, 255, 200, 0.4), 0 0 60px rgba(0, 255, 200, 0.2)',
                'glow-fake': '0 0 30px rgba(255, 0, 102, 0.4), 0 0 60px rgba(255, 0, 102, 0.2)',
                'card': '0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.06)',
                'card-hover': '0 12px 40px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(0, 240, 255, 0.2)',
            },
            animation: {
                'pulse-slow': 'pulse-slow 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
                'gradient-shift': 'gradient-shift 8s ease infinite',
            },
            keyframes: {
                'pulse-slow': {
                    '0%, 100%': { opacity: '0.4', transform: 'scale(1)' },
                    '50%': { opacity: '0.7', transform: 'scale(1.02)' },
                },
                'glow-pulse': {
                    '0%, 100%': { boxShadow: '0 0 20px rgba(0, 240, 255, 0.3), 0 0 40px rgba(255, 0, 212, 0.2)' },
                    '50%': { boxShadow: '0 0 30px rgba(0, 240, 255, 0.5), 0 0 60px rgba(255, 0, 212, 0.3)' },
                },
                'gradient-shift': {
                    '0%': { backgroundPosition: '0% 50%' },
                    '50%': { backgroundPosition: '100% 50%' },
                    '100%': { backgroundPosition: '0% 50%' },
                },
            },
        },
    },
    plugins: [],
};

export default config;
