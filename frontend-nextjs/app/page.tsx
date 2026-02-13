'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import {
    Scan,
    Brain,
    Sparkles,
    ArrowRight,
    CheckCircle2,
    Zap,
    Lock,
    BarChart3,
    PlayCircle,
    Shield,
    Code2,
    Database,
    Layers,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ROUTES } from '@/lib/constants';

const features = [
    {
        icon: Brain,
        title: 'Advanced AI Models',
        description:
            'EfficientNet B4 with Auto-Attention mechanism, trained on 100K+ videos from DFDC and FaceForensics++ datasets. Achieves 94% accuracy with ensemble CNN architecture.',
        color: 'from-accent-cyan to-accent-purple',
        iconColor: 'text-accent-cyan',
    },
    {
        icon: Scan,
        title: 'Real-Time Detection',
        description:
            'GPU-accelerated BlazeFace for face detection, WebSocket-based live progress updates, and PyTorch inference pipeline for sub-3-second analysis.',
        color: 'from-accent-magenta to-accent-pink',
        iconColor: 'text-accent-magenta',
    },
    {
        icon: BarChart3,
        title: 'Detailed Analytics',
        description:
            'Frame-by-frame confidence scoring with temporal analysis, multi-model comparison dashboard, and exportable JSON/CSV reports with visualization.',
        color: 'from-accent-purple to-accent-cyan',
        iconColor: 'text-accent-purple',
    },
    {
        icon: Lock,
        title: 'Privacy-First Architecture',
        description:
            'Client-side file handling, automatic server-side cleanup post-analysis, no data persistence, and CORS-protected REST API with rate limiting.',
        color: 'from-neon-lime to-accent-cyan',
        iconColor: 'text-neon-lime',
    },
];

const techStack = [
    {
        category: 'Frontend',
        icon: Code2,
        color: 'from-accent-cyan to-accent-purple',
        iconColor: 'text-accent-cyan',
        technologies: ['Next.js 15', 'React 18', 'TypeScript', 'Tailwind CSS', 'Framer Motion', 'Socket.IO Client'],
    },
    {
        category: 'Backend',
        icon: Database,
        color: 'from-accent-magenta to-accent-pink',
        iconColor: 'text-accent-magenta',
        technologies: ['Flask', 'Flask-SocketIO', 'Python 3.10+', 'OpenCV', 'NumPy', 'SciPy'],
    },
    {
        category: 'ML/AI',
        icon: Layers,
        color: 'from-neon-lime to-accent-cyan',
        iconColor: 'text-neon-lime',
        technologies: ['PyTorch 2.0', 'EfficientNet', 'BlazeFace', 'DFDC Dataset', 'FaceForensics++', 'Auto-Attention'],
    },
];

const stats = [
    { value: '94%', label: 'Detection Accuracy', color: 'text-accent-cyan' },
    { value: '100K+', label: 'Training Videos', color: 'text-accent-magenta' },
    { value: '<3s', label: 'Avg Processing Time', color: 'text-accent-purple' },
    { value: '4', label: 'Model Variants', color: 'text-neon-lime' },
];

export default function HomePage() {
    return (
        <div className="overflow-hidden">
            {/* Hero Section */}
            <section className="relative min-h-screen flex items-center pt-20">
                {/* Animated Aurora Background */}
                <div className="absolute inset-0 overflow-hidden">
                    {/* Floating orbs */}
                    <div className="orb-cyan" style={{ top: '10%', left: '10%' }} />
                    <div className="orb-magenta" style={{ top: '30%', right: '15%' }} />
                    <div className="orb-purple" style={{ bottom: '20%', left: '20%' }} />
                    <div className="orb-lime" style={{ bottom: '30%', right: '25%' }} />

                    {/* Grid pattern */}
                    <div
                        className="absolute inset-0 opacity-[0.03]"
                        style={{
                            backgroundImage: `
                linear-gradient(rgba(0, 240, 255, 0.3) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 240, 255, 0.3) 1px, transparent 1px)
              `,
                            backgroundSize: '50px 50px',
                        }}
                    />
                </div>

                <div className="container-custom relative z-10">
                    <div className="max-w-4xl mx-auto text-center">
                        {/* Badge */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 }}
                        >
                            <Badge variant="default" className="mb-6 py-2 px-4 animate-glow-pulse">
                                <Sparkles className="w-4 h-4 mr-2 text-accent-cyan" />
                                <span className="text-gradient-aurora">AI-Powered Detection Technology</span>
                            </Badge>
                        </motion.div>

                        {/* Headline */}
                        <motion.h1
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            className="text-4xl md:text-6xl lg:text-7xl font-bold text-text-primary mb-6 leading-tight font-display"
                        >
                            Detect AI-Generated Faces with{' '}
                            <span className="text-gradient-aurora inline-block">
                                Cutting-Edge Deep Learning
                            </span>
                        </motion.h1>

                        {/* Subtitle */}
                        <motion.p
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 }}
                            className="text-lg md:text-xl text-text-secondary mb-8 max-w-2xl mx-auto"
                        >
                            Advanced deepfake detection system powered by EfficientNet ensemble CNNs, trained on 100,000+ videos.
                            Real-time WebSocket analysis with GPU acceleration and frame-by-frame temporal verification.
                        </motion.p>

                        {/* CTA Buttons */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.4 }}
                            className="flex flex-col sm:flex-row items-center justify-center gap-4"
                        >
                            <Link href={ROUTES.ANALYZE}>
                                <Button size="xl" className="group">
                                    <Shield className="w-5 h-5 mr-2" />
                                    Start Detection
                                    <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                                </Button>
                            </Link>
                            <Link href={ROUTES.HOW_IT_WORKS}>
                                <Button variant="secondary" size="xl">
                                    <PlayCircle className="w-5 h-5 mr-2" />
                                    See How It Works
                                </Button>
                            </Link>
                        </motion.div>

                        {/* Trust indicators */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.6 }}
                            className="mt-12 flex flex-wrap items-center justify-center gap-6 text-sm text-text-muted"
                        >
                            {[
                                { text: 'No registration required', icon: CheckCircle2, color: 'text-success-400' },
                                { text: 'Free to use', icon: Zap, color: 'text-neon-yellow' },
                                { text: 'Private & Secure', icon: Lock, color: 'text-accent-cyan' },
                            ].map((item) => (
                                <span key={item.text} className="flex items-center gap-2">
                                    <item.icon className={`w-4 h-4 ${item.color}`} />
                                    {item.text}
                                </span>
                            ))}
                        </motion.div>
                    </div>
                </div>

                {/* Scroll indicator */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1 }}
                    className="absolute bottom-8 left-1/2 -translate-x-1/2"
                >
                    <motion.div
                        animate={{ y: [0, 10, 0] }}
                        transition={{ repeat: Infinity, duration: 1.5 }}
                        className="w-6 h-10 rounded-full border-2 border-accent-cyan/50 flex items-start justify-center p-2"
                    >
                        <div className="w-1 h-2 rounded-full bg-accent-cyan" />
                    </motion.div>
                </motion.div>
            </section>

            {/* Stats Section */}
            <section className="py-16 border-y border-accent-cyan/10 bg-dark-deep/50 relative">
                {/* Subtle gradient background */}
                <div className="absolute inset-0 bg-gradient-to-r from-accent-cyan/5 via-transparent to-accent-magenta/5" />

                <div className="container-custom relative">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                        {stats.map((stat, index) => (
                            <motion.div
                                key={stat.label}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.1 }}
                                className="text-center"
                            >
                                <div
                                    className={`text-4xl md:text-5xl font-bold ${stat.color} mb-2 font-display`}
                                    style={{ textShadow: '0 0 30px currentColor' }}
                                >
                                    {stat.value}
                                </div>
                                <div className="text-sm text-text-secondary">{stat.label}</div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="py-24 relative">
                <div className="container-custom">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="text-center mb-16"
                    >
                        <Badge variant="magenta" className="mb-4">
                            <Sparkles className="w-3 h-3" />
                            Features
                        </Badge>
                        <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4 font-display">
                            Why Choose{' '}
                            <span className="text-gradient-primary">DeepFakeHub</span>?
                        </h2>
                        <p className="text-text-secondary max-w-2xl mx-auto">
                            Our platform combines the latest in AI research with an intuitive
                            interface to make deepfake detection accessible to everyone.
                        </p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {features.map((feature, index) => (
                            <motion.div
                                key={feature.title}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.1 }}
                            >
                                <Card variant="glass" hover className="h-full">
                                    <CardContent className="p-6">
                                        <div
                                            className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${feature.color} p-[1px] mb-4`}
                                        >
                                            <div className="w-full h-full rounded-2xl bg-dark-card flex items-center justify-center">
                                                <feature.icon className={`w-7 h-7 ${feature.iconColor}`} />
                                            </div>
                                        </div>
                                        <h3 className="text-xl font-semibold text-text-primary mb-2 font-display">
                                            {feature.title}
                                        </h3>
                                        <p className="text-text-secondary">{feature.description}</p>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* How it works - Brief */}
            <section className="py-24 bg-dark-secondary/30 relative overflow-hidden">
                {/* Background decoration */}
                <div className="absolute top-0 right-0 w-96 h-96 bg-accent-magenta/10 rounded-full blur-3xl" />
                <div className="absolute bottom-0 left-0 w-96 h-96 bg-accent-cyan/10 rounded-full blur-3xl" />

                <div className="container-custom relative">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="text-center mb-16"
                    >
                        <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4 font-display">
                            How It <span className="text-gradient-aurora">Works</span>
                        </h2>
                        <p className="text-text-secondary max-w-2xl mx-auto">
                            Three simple steps to detect deepfakes in any media
                        </p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        {[
                            {
                                step: '01',
                                title: 'Upload Media',
                                description: 'Drag and drop or select an image or video file',
                                gradient: 'from-accent-cyan to-accent-purple',
                                glow: 'shadow-glow-cyan',
                            },
                            {
                                step: '02',
                                title: 'AI Analysis',
                                description: 'Our models analyze faces for manipulation signs',
                                gradient: 'from-accent-magenta to-accent-pink',
                                glow: 'shadow-glow-magenta',
                            },
                            {
                                step: '03',
                                title: 'Get Results',
                                description: 'Receive detailed verdict with confidence scores',
                                gradient: 'from-neon-lime to-accent-cyan',
                                glow: 'shadow-glow-real',
                            },
                        ].map((item, index) => (
                            <motion.div
                                key={item.step}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.15 }}
                                className="relative"
                            >
                                <div className="text-center">
                                    {/* Step number with gradient ring */}
                                    <div
                                        className={`w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br ${item.gradient} p-[2px] ${item.glow}`}
                                    >
                                        <div className="w-full h-full rounded-full bg-dark-primary flex items-center justify-center">
                                            <span
                                                className={`text-2xl font-bold font-display bg-gradient-to-br ${item.gradient} bg-clip-text text-transparent`}
                                            >
                                                {item.step}
                                            </span>
                                        </div>
                                    </div>

                                    <h3 className="text-xl font-semibold text-text-primary mb-2 font-display">
                                        {item.title}
                                    </h3>
                                    <p className="text-text-secondary">{item.description}</p>
                                </div>

                                {/* Connector line */}
                                {index < 2 && (
                                    <div className="hidden md:block absolute top-10 left-[60%] w-[80%] h-px">
                                        <div className="w-full h-px bg-gradient-to-r from-accent-cyan/50 via-accent-magenta/30 to-transparent" />
                                    </div>
                                )}
                            </motion.div>
                        ))}
                    </div>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="text-center mt-12"
                    >
                        <Link href={ROUTES.ANALYZE}>
                            <Button size="lg" className="animate-glow-pulse">
                                Try It Now <ArrowRight className="w-5 h-5 ml-2" />
                            </Button>
                        </Link>
                    </motion.div>
                </div>
            </section>

            {/* Technical Stack Section - NEW */}
            <section className="py-24 relative">
                <div className="container-custom">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="text-center mb-16"
                    >
                        <Badge variant="cyan" className="mb-4">
                            <Code2 className="w-3 h-3" />
                            Technical Stack
                        </Badge>
                        <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4 font-display">
                            Built with{' '}
                            <span className="text-gradient-primary">Modern Technologies</span>
                        </h2>
                        <p className="text-text-secondary max-w-2xl mx-auto">
                            Full-stack application leveraging cutting-edge frameworks and ML libraries for production-grade deepfake detection.
                        </p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {techStack.map((stack, index) => (
                            <motion.div
                                key={stack.category}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.1 }}
                            >
                                <Card variant="glass" hover className="h-full">
                                    <CardContent className="p-6">
                                        <div
                                            className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${stack.color} p-[1px] mb-4`}
                                        >
                                            <div className="w-full h-full rounded-2xl bg-dark-card flex items-center justify-center">
                                                <stack.icon className={`w-7 h-7 ${stack.iconColor}`} />
                                            </div>
                                        </div>
                                        <h3 className="text-xl font-semibold text-text-primary mb-4 font-display">
                                            {stack.category}
                                        </h3>
                                        <div className="flex flex-wrap gap-2">
                                            {stack.technologies.map((tech) => (
                                                <Badge
                                                    key={tech}
                                                    variant="outline"
                                                    className="text-xs border-dark-border text-text-secondary hover:border-accent-cyan/50 hover:text-accent-cyan transition-colors"
                                                >
                                                    {tech}
                                                </Badge>
                                            ))}
                                        </div>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="py-24">
                <div className="container-custom">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        className="relative overflow-hidden rounded-3xl p-8 md:p-16"
                    >
                        {/* Animated gradient background */}
                        <div className="absolute inset-0 bg-gradient-aurora bg-[length:300%_300%] animate-gradient-shift" />

                        {/* Overlay pattern */}
                        <div
                            className="absolute inset-0 opacity-10"
                            style={{
                                backgroundImage: `
                  radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%),
                  radial-gradient(circle at 80% 50%, rgba(255,255,255,0.1) 0%, transparent 50%)
                `,
                            }}
                        />

                        <div className="relative z-10 text-center">
                            <motion.div
                                initial={{ scale: 0 }}
                                whileInView={{ scale: 1 }}
                                viewport={{ once: true }}
                                className="w-20 h-20 mx-auto mb-6 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center"
                            >
                                <Shield className="w-10 h-10 text-white" />
                            </motion.div>

                            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4 font-display">
                                Ready to Detect Deepfakes?
                            </h2>
                            <p className="text-white/80 max-w-xl mx-auto mb-8">
                                Start analyzing your media now with our free, privacy-focused
                                deepfake detection tool. No sign-up required.
                            </p>
                            <Link href={ROUTES.ANALYZE}>
                                <Button
                                    variant="secondary"
                                    size="xl"
                                    className="bg-white text-dark-void hover:bg-white/90 shadow-2xl"
                                >
                                    <Zap className="w-5 h-5 mr-2" />
                                    Begin Analysis
                                </Button>
                            </Link>
                        </div>
                    </motion.div>
                </div>
            </section>
        </div>
    );
}
