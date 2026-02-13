import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Analyze Media',
    description: 'Upload and analyze images or videos for deepfake detection using advanced AI models.',
};

export default function AnalyzeLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return children;
}
