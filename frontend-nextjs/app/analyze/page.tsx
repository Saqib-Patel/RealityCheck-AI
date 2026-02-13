'use client';

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, ArrowLeft, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { FileUploadZone } from '@/components/features/file-upload-zone';
import { ModelSelector } from '@/components/features/model-selector';
import { AnalysisProgress } from '@/components/features/analysis-progress';
import { AnalysisResultCard } from '@/components/features/analysis-result-card';
import { useAnalysis } from '@/hooks/use-analysis';
import { DEFAULT_MODEL, DEFAULT_DATASET } from '@/lib/constants';
import type { AnalysisResult } from '@/types/analysis';

export default function AnalyzePage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
    const [selectedDataset, setSelectedDataset] = useState(DEFAULT_DATASET);
    const { analyze, reset, status, isAnalyzing, uploadProgress, result, error } = useAnalysis();

    const handleFileSelect = useCallback((file: File) => {
        setSelectedFile(file);
        reset();
    }, [reset]);

    const handleClearFile = useCallback(() => {
        setSelectedFile(null);
        reset();
    }, [reset]);

    const handleAnalyze = async () => {
        if (!selectedFile) return;

        try {
            await analyze(selectedFile, selectedModel, selectedDataset);
        } catch (err) {
            console.error('Analysis failed:', err);
        }
    };

    const handleNewAnalysis = () => {
        setSelectedFile(null);
        reset();
    };

    return (
        <div className="min-h-screen py-8 md:py-12">
            <div className="container-custom max-w-5xl">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center mb-8 md:mb-12"
                >
                    <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-accent-cyan/20 to-accent-magenta/20 flex items-center justify-center border border-accent-cyan/30">
                        <Shield className="w-8 h-8 text-accent-cyan" />
                    </div>
                    <h1 className="text-3xl md:text-4xl font-bold text-text-primary mb-2 font-display">
                        Analyze Media
                    </h1>
                    <p className="text-text-secondary max-w-lg mx-auto">
                        Upload an image or video to detect potential deepfake manipulation using AI
                    </p>
                </motion.div>

                <AnimatePresence mode="wait">
                    {/* Result View */}
                    {result && status === 'complete' && (
                        <motion.div
                            key="result"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                        >
                            <AnalysisResultCard result={result} />

                            <div className="flex justify-center mt-8">
                                <Button onClick={handleNewAnalysis} size="lg">
                                    <RefreshCw className="w-5 h-5 mr-2" />
                                    Analyze Another File
                                </Button>
                            </div>
                        </motion.div>
                    )}

                    {/* Analysis in Progress */}
                    {isAnalyzing && (
                        <motion.div
                            key="progress"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                        >
                            <AnalysisProgress
                                progress={uploadProgress}
                                status={status === 'uploading' ? 'uploading' : 'analyzing'}
                                currentStep={status === 'uploading' ? 'Uploading file...' : 'Analyzing for deepfakes...'}
                            />
                        </motion.div>
                    )}

                    {/* Upload & Configure View */}
                    {!result && !isAnalyzing && (
                        <motion.div
                            key="upload"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="space-y-6"
                        >
                            {/* File Upload */}
                            <FileUploadZone
                                selectedFile={selectedFile}
                                onFileSelect={handleFileSelect}
                                onClear={handleClearFile}
                                isUploading={status === 'uploading'}
                                uploadProgress={uploadProgress}
                            />

                            {/* Model Configuration */}
                            {selectedFile && (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.1 }}
                                >
                                    <ModelSelector
                                        selectedModel={selectedModel}
                                        selectedDataset={selectedDataset}
                                        onModelChange={setSelectedModel}
                                        onDatasetChange={setSelectedDataset}
                                    />
                                </motion.div>
                            )}

                            {/* Error Message */}
                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, y: -10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="p-4 rounded-xl bg-danger-400/10 border border-danger-400/30 text-danger-400"
                                >
                                    <p className="font-medium">Analysis Failed</p>
                                    <p className="text-sm opacity-80">{error}</p>
                                </motion.div>
                            )}

                            {/* Analyze Button */}
                            {selectedFile && (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.2 }}
                                    className="flex justify-center"
                                >
                                    <Button
                                        size="xl"
                                        onClick={handleAnalyze}
                                        disabled={!selectedFile || isAnalyzing}
                                        className="min-w-[200px]"
                                    >
                                        <Shield className="w-5 h-5 mr-2" />
                                        Analyze File
                                    </Button>
                                </motion.div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
