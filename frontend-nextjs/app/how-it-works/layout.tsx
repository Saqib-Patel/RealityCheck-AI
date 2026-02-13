import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'How It Works',
    description: 'Learn how our AI-powered deepfake detection system works using state-of-the-art deep learning models.',
};

export default function HowItWorksLayout({ children }: { children: React.ReactNode }) {
    return children;
}
