'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Upload, Scan, Brain, BarChart3, Shield, ArrowRight, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ROUTES, MODELS, DATASETS } from '@/lib/constants';

const steps = [
    {
        icon: Upload,
        title: 'Upload Your Media',
        description: 'Drag and drop or select any image (JPG, PNG, WebP) or video (MP4, AVI, MOV) file up to 500MB.',
        gradient: 'from-accent-cyan to-accent-purple',
        details: ['Supports high-resolution media', 'Automatic format detection', 'Secure local processing'],
    },
    {
        icon: Scan,
        title: 'Face Detection',
        description: 'Our BlazeFace model rapidly detects and extracts all faces from your media for analysis.',
        gradient: 'from-accent-purple to-accent-magenta',
        details: ['Multi-face detection', 'Precise face cropping', 'Frame-by-frame for videos'],
    },
    {
        icon: Brain,
        title: 'AI Analysis',
        description: 'EfficientNet with Auto-Attention analyzes facial features for signs of AI manipulation.',
        gradient: 'from-accent-magenta to-neon-orange',
        details: ['Deep learning analysis', 'Attention mechanism for accuracy', 'Trained on 100K+ samples'],
    },
    {
        icon: BarChart3,
        title: 'Get Results',
        description: 'Receive a detailed report with confidence scores, frame analysis, and exportable data.',
        gradient: 'from-neon-lime to-accent-cyan',
        details: ['Clear verdict display', 'Frame-by-frame breakdown', 'JSON/CSV export'],
    },
];

export default function HowItWorksPage() {
    return (
        <div className="min-h-screen py-12">
            <div className="container-custom max-w-5xl">
                {/* Header */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-16">
                    <Badge variant="purple" className="mb-4">Learn More</Badge>
                    <h1 className="text-4xl md:text-5xl font-bold text-text-primary mb-4 font-display">
                        How <span className="text-gradient-aurora">DeepFake Detection</span> Works
                    </h1>
                    <p className="text-text-secondary max-w-2xl mx-auto text-lg">
                        Our AI-powered system uses state-of-the-art deep learning to identify manipulated media with high accuracy.
                    </p>
                </motion.div>

                {/* Steps */}
                <div className="space-y-8 mb-16">
                    {steps.map((step, index) => (
                        <motion.div
                            key={step.title}
                            initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <Card variant="glass" hover>
                                <CardContent className="p-6 md:p-8">
                                    <div className="flex flex-col md:flex-row gap-6 items-start">
                                        <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${step.gradient} p-[2px] flex-shrink-0`}>
                                            <div className="w-full h-full rounded-2xl bg-dark-card flex items-center justify-center">
                                                <step.icon className="w-8 h-8 text-white" />
                                            </div>
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex items-center gap-3 mb-2">
                                                <span className={`text-sm font-bold bg-gradient-to-r ${step.gradient} bg-clip-text text-transparent`}>
                                                    Step {index + 1}
                                                </span>
                                            </div>
                                            <h3 className="text-xl font-bold text-text-primary mb-2 font-display">{step.title}</h3>
                                            <p className="text-text-secondary mb-4">{step.description}</p>
                                            <div className="flex flex-wrap gap-2">
                                                {step.details.map((detail) => (
                                                    <span key={detail} className="flex items-center gap-1 text-sm text-text-muted">
                                                        <CheckCircle2 className="w-4 h-4 text-success-400" />
                                                        {detail}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </motion.div>
                    ))}
                </div>

                {/* Models Section */}
                <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="mb-16">
                    <h2 className="text-2xl font-bold text-text-primary mb-6 font-display text-center">Available Models</h2>
                    <div className="grid md:grid-cols-2 gap-4">
                        {MODELS.map((model) => (
                            <Card key={model.name} variant="default" hover>
                                <CardContent className="p-4">
                                    <div className="flex justify-between items-start mb-2">
                                        <h4 className="font-semibold text-text-primary">{model.name}</h4>
                                        <Badge variant={model.speed === 'Fast' ? 'cyan' : 'purple'}>{model.speed}</Badge>
                                    </div>
                                    <p className="text-sm text-text-secondary mb-2">{model.description}</p>
                                    <div className="text-sm text-success-400">{model.accuracy}% accuracy</div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </motion.div>

                {/* CTA */}
                <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center">
                    <Card className="p-8 bg-gradient-to-r from-accent-cyan/10 to-accent-magenta/10 border-accent-cyan/30">
                        <Shield className="w-12 h-12 mx-auto mb-4 text-accent-cyan" />
                        <h3 className="text-xl font-bold text-text-primary mb-2 font-display">Ready to Try?</h3>
                        <p className="text-text-secondary mb-6">Upload your first file and see the AI in action.</p>
                        <Link href={ROUTES.ANALYZE}>
                            <Button size="lg">
                                Start Analyzing <ArrowRight className="w-5 h-5 ml-2" />
                            </Button>
                        </Link>
                    </Card>
                </motion.div>
            </div>
        </div>
    );
}
