import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Analysis History',
    description: 'View your previous deepfake detection analysis results.',
};

export default function HistoryLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return children;
}
