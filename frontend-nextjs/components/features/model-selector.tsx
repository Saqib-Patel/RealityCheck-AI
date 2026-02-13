'use client';

import { Brain, Zap, Database, ChevronRight } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils/cn';
import { MODELS, DATASETS, DEFAULT_MODEL, DEFAULT_DATASET } from '@/lib/constants';

interface ModelSelectorProps {
    selectedModel: string;
    selectedDataset: string;
    onModelChange: (model: string) => void;
    onDatasetChange: (dataset: string) => void;
    className?: string;
}

export function ModelSelector({
    selectedModel = DEFAULT_MODEL,
    selectedDataset = DEFAULT_DATASET,
    onModelChange,
    onDatasetChange,
    className,
}: ModelSelectorProps) {
    const selectedModelData = MODELS.find((m) => m.name === selectedModel);

    return (
        <Card variant="glass" className={cn('', className)}>
            <CardContent className="p-6">
                <div className="flex items-center gap-2 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-purple/20 to-accent-magenta/20 flex items-center justify-center">
                        <Brain className="w-5 h-5 text-accent-purple" />
                    </div>
                    <div>
                        <h3 className="text-lg font-semibold text-text-primary font-display">
                            AI Model Configuration
                        </h3>
                        <p className="text-sm text-text-muted">
                            Select the model and training dataset
                        </p>
                    </div>
                </div>

                <div className="space-y-4">
                    {/* Model Selection */}
                    <div>
                        <label className="block text-sm font-medium text-text-secondary mb-2">
                            Detection Model
                        </label>
                        <Select value={selectedModel} onValueChange={onModelChange}>
                            <SelectTrigger>
                                <SelectValue placeholder="Select a model" />
                            </SelectTrigger>
                            <SelectContent>
                                {MODELS.map((model) => (
                                    <SelectItem key={model.name} value={model.name}>
                                        <div className="flex items-center justify-between w-full">
                                            <span>{model.name}</span>
                                            <Badge
                                                variant={
                                                    model.speed === 'Fast'
                                                        ? 'cyan'
                                                        : model.speed === 'Medium'
                                                            ? 'purple'
                                                            : 'magenta'
                                                }
                                                className="ml-2"
                                            >
                                                {model.speed}
                                            </Badge>
                                        </div>
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>

                        {/* Model details */}
                        {selectedModelData && (
                            <div className="mt-3 p-3 rounded-lg bg-dark-hover/50 border border-dark-border">
                                <p className="text-sm text-text-secondary mb-2">
                                    {selectedModelData.description}
                                </p>
                                <div className="flex items-center gap-4 text-xs">
                                    <span className="flex items-center gap-1 text-success-400">
                                        <Zap className="w-3.5 h-3.5" />
                                        {selectedModelData.accuracy}% Accuracy
                                    </span>
                                    <span className="flex items-center gap-1 text-text-muted">
                                        <ChevronRight className="w-3.5 h-3.5" />
                                        {selectedModelData.speed} Processing
                                    </span>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Dataset Selection */}
                    <div>
                        <label className="block text-sm font-medium text-text-secondary mb-2">
                            Training Dataset
                        </label>
                        <Select value={selectedDataset} onValueChange={onDatasetChange}>
                            <SelectTrigger>
                                <SelectValue placeholder="Select a dataset" />
                            </SelectTrigger>
                            <SelectContent>
                                {DATASETS.map((dataset) => (
                                    <SelectItem key={dataset.name} value={dataset.name}>
                                        <div className="flex items-center gap-2">
                                            <Database className="w-4 h-4 text-accent-cyan" />
                                            <span>{dataset.name}</span>
                                            <span className="text-xs text-text-muted">
                                                ({dataset.videoCount?.toLocaleString()} videos)
                                            </span>
                                        </div>
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                </div>

                {/* Recommended badge */}
                {selectedModel === 'EfficientNetAutoAttB4' && (
                    <div className="mt-4 flex items-center gap-2 text-sm text-accent-cyan">
                        <Badge variant="solid" className="text-xs">Recommended</Badge>
                        <span className="text-text-muted">Best balance of accuracy and speed</span>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
