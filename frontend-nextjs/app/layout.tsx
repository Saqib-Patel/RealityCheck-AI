import type { Metadata } from 'next';
import { Inter, Space_Grotesk } from 'next/font/google';
import './globals.css';
import { Header } from '@/components/layout/header';
import { Footer } from '@/components/layout/footer';
import { TooltipProvider } from '@/components/ui/tooltip';
import { ServerStatus } from '@/components/server-status';

const inter = Inter({
    subsets: ['latin'],
    variable: '--font-inter',
    display: 'swap',
});

const spaceGrotesk = Space_Grotesk({
    subsets: ['latin'],
    variable: '--font-space-grotesk',
    display: 'swap',
});

export const metadata: Metadata = {
    metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'https://deepfakehub.com'),
    title: {
        default: 'DeepFake Detection Hub | AI-Powered Deepfake Detection',
        template: '%s | DeepFake Hub',
    },
    description:
        'Advanced deepfake detection system using EfficientNet ensemble CNNs, trained on 100K+ videos. Real-time WebSocket analysis with GPU acceleration. Built by Mohammed Saqib Patel.',
    keywords: [
        'deepfake detection',
        'AI detection',
        'face analysis',
        'deep learning',
        'machine learning',
        'PyTorch',
        'EfficientNet',
        'computer vision',
        'video analysis',
        'image analysis',
        'fake detection',
        'CNN',
        'neural networks',
        'BlazeFace',
        'DFDC',
        'FaceForensics++',
    ],
    authors: [{ name: 'Mohammed Saqib Patel' }],
    openGraph: {
        type: 'website',
        locale: 'en_US',
        url: 'https://deepfakehub.com',
        siteName: 'DeepFake Detection Hub',
        title: 'DeepFake Detection Hub | AI-Powered Deepfake Detection',
        description:
            'Detect AI-generated faces and deepfakes with cutting-edge deep learning models.',
        images: [
            {
                url: '/og-image.png',
                width: 1200,
                height: 630,
                alt: 'DeepFake Detection Hub',
            },
        ],
    },
    twitter: {
        card: 'summary_large_image',
        title: 'DeepFake Detection Hub',
        description: 'AI-powered deepfake detection using state-of-the-art models.',
        images: ['/og-image.png'],
    },
    robots: {
        index: true,
        follow: true,
        googleBot: {
            index: true,
            follow: true,
            'max-video-preview': -1,
            'max-image-preview': 'large',
            'max-snippet': -1,
        },
    },
    icons: {
        icon: '/favicon.ico',
        shortcut: '/favicon-16x16.png',
        apple: '/apple-touch-icon.png',
    },
    manifest: '/site.webmanifest',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className={`${inter.variable} ${spaceGrotesk.variable}`}>
            <body className="min-h-screen bg-dark-void antialiased">
                <TooltipProvider>
                    <ServerStatus />
                    <div className="flex min-h-screen flex-col">
                        <Header />
                        <main className="flex-1 pt-16 md:pt-20">{children}</main>
                        <Footer />
                    </div>
                </TooltipProvider>
            </body>
        </html>
    );
}
