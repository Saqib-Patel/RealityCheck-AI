'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, Image as ImageIcon, Film, FileWarning, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import { formatFileSize } from '@/lib/utils/format';
import { validateFile, isImageFile, isVideoFile } from '@/lib/utils/validation';
import { Button } from '@/components/ui/button';

interface FileUploadZoneProps {
    onFileSelect: (file: File) => void;
    onClear: () => void;
    selectedFile: File | null;
    isUploading?: boolean;
    uploadProgress?: number;
    disabled?: boolean;
    className?: string;
}

export function FileUploadZone({
    onFileSelect,
    onClear,
    selectedFile,
    isUploading = false,
    uploadProgress = 0,
    disabled = false,
    className,
}: FileUploadZoneProps) {
    const [error, setError] = useState<string | null>(null);
    const [preview, setPreview] = useState<string | null>(null);

    const onDrop = useCallback(
        (acceptedFiles: File[]) => {
            if (acceptedFiles.length === 0) return;

            const file = acceptedFiles[0];
            const validation = validateFile(file);

            if (!validation.isValid) {
                setError(validation.error || 'Invalid file');
                return;
            }

            setError(null);

            // Generate preview for images
            if (isImageFile(file)) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    setPreview(e.target?.result as string);
                };
                reader.readAsDataURL(file);
            } else if (isVideoFile(file)) {
                // Generate video thumbnail
                const video = document.createElement('video');
                video.preload = 'metadata';
                video.onloadeddata = () => {
                    video.currentTime = 1; // Seek to 1 second
                };
                video.onseeked = () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
                    setPreview(canvas.toDataURL('image/jpeg'));
                };
                video.src = URL.createObjectURL(file);
            }

            onFileSelect(file);
        },
        [onFileSelect]
    );

    const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
        onDrop,
        accept: {
            'image/jpeg': ['.jpg', '.jpeg'],
            'image/png': ['.png'],
            'image/webp': ['.webp'],
            'video/mp4': ['.mp4'],
            'video/avi': ['.avi'],
            'video/quicktime': ['.mov'],
        },
        maxFiles: 1,
        disabled: disabled || isUploading,
    });

    const handleClear = () => {
        setPreview(null);
        setError(null);
        onClear();
    };

    return (
        <div className={cn('w-full', className)}>
            <AnimatePresence mode="wait">
                {!selectedFile ? (
                    <motion.div
                        key="dropzone"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                    >
                        <div
                            {...getRootProps()}
                            className={cn(
                                'relative rounded-2xl border-2 border-dashed p-8 md:p-12 transition-all duration-300 cursor-pointer',
                                'bg-dark-elevated/50 backdrop-blur-sm',
                                isDragActive && !isDragReject && 'border-accent-cyan bg-accent-cyan/5 shadow-glow-cyan',
                                isDragReject && 'border-danger-400 bg-danger-400/5',
                                !isDragActive && !isDragReject && 'border-dark-border hover:border-accent-cyan/50 hover:bg-dark-hover/50',
                                disabled && 'opacity-50 cursor-not-allowed'
                            )}
                        >
                            <input {...getInputProps()} />

                            {/* Animated background orbs */}
                            <div className="absolute inset-0 overflow-hidden rounded-2xl pointer-events-none">
                                <div className="absolute top-1/4 left-1/4 w-32 h-32 bg-accent-cyan/10 rounded-full blur-3xl animate-pulse-slow" />
                                <div className="absolute bottom-1/4 right-1/4 w-24 h-24 bg-accent-magenta/10 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }} />
                            </div>

                            <div className="relative flex flex-col items-center text-center">
                                {/* Upload Icon */}
                                <motion.div
                                    animate={{ y: isDragActive ? -10 : 0 }}
                                    className={cn(
                                        'w-20 h-20 rounded-2xl flex items-center justify-center mb-6',
                                        'bg-gradient-to-br from-accent-cyan/20 to-accent-magenta/20',
                                        'border border-accent-cyan/30'
                                    )}
                                >
                                    <Upload
                                        className={cn(
                                            'w-8 h-8 transition-colors duration-200',
                                            isDragActive ? 'text-accent-cyan' : 'text-text-secondary'
                                        )}
                                    />
                                </motion.div>

                                {/* Text */}
                                <h3 className="text-xl font-semibold text-text-primary mb-2 font-display">
                                    {isDragActive ? 'Drop your file here' : 'Drag & drop your media'}
                                </h3>
                                <p className="text-text-secondary mb-4">
                                    or click to browse from your computer
                                </p>

                                {/* File type hints */}
                                <div className="flex items-center gap-4 text-sm text-text-muted">
                                    <span className="flex items-center gap-1.5">
                                        <ImageIcon className="w-4 h-4" />
                                        JPG, PNG, WebP
                                    </span>
                                    <span className="w-px h-4 bg-dark-border" />
                                    <span className="flex items-center gap-1.5">
                                        <Film className="w-4 h-4" />
                                        MP4, AVI, MOV
                                    </span>
                                </div>

                                <p className="text-xs text-text-dim mt-4">Maximum file size: 5MB</p>
                            </div>
                        </div>
                    </motion.div>
                ) : (
                    <motion.div
                        key="preview"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="relative rounded-2xl border border-dark-border bg-dark-elevated overflow-hidden"
                    >
                        {/* Preview */}
                        <div className="relative aspect-video bg-dark-card">
                            {preview ? (
                                <img
                                    src={preview}
                                    alt="Preview"
                                    className="w-full h-full object-contain"
                                />
                            ) : (
                                <div className="w-full h-full flex items-center justify-center">
                                    {isVideoFile(selectedFile) ? (
                                        <Film className="w-16 h-16 text-text-muted" />
                                    ) : (
                                        <ImageIcon className="w-16 h-16 text-text-muted" />
                                    )}
                                </div>
                            )}

                            {/* Upload progress overlay */}
                            {isUploading && (
                                <div className="absolute inset-0 bg-dark-void/80 flex flex-col items-center justify-center">
                                    <Loader2 className="w-10 h-10 text-accent-cyan animate-spin mb-4" />
                                    <div className="w-48 h-2 bg-dark-elevated rounded-full overflow-hidden">
                                        <motion.div
                                            className="h-full bg-gradient-to-r from-accent-cyan to-accent-magenta"
                                            initial={{ width: 0 }}
                                            animate={{ width: `${uploadProgress}%` }}
                                            transition={{ duration: 0.3 }}
                                        />
                                    </div>
                                    <p className="text-sm text-text-secondary mt-2">
                                        Uploading... {uploadProgress}%
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* File info */}
                        <div className="p-4 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className={cn(
                                    'w-10 h-10 rounded-lg flex items-center justify-center',
                                    isVideoFile(selectedFile) ? 'bg-accent-magenta/10' : 'bg-accent-cyan/10'
                                )}>
                                    {isVideoFile(selectedFile) ? (
                                        <Film className="w-5 h-5 text-accent-magenta" />
                                    ) : (
                                        <ImageIcon className="w-5 h-5 text-accent-cyan" />
                                    )}
                                </div>
                                <div>
                                    <p className="text-sm font-medium text-text-primary truncate max-w-xs">
                                        {selectedFile.name}
                                    </p>
                                    <p className="text-xs text-text-muted">
                                        {formatFileSize(selectedFile.size)} • {isVideoFile(selectedFile) ? 'Video' : 'Image'}
                                    </p>
                                </div>
                            </div>

                            {!isUploading && (
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={handleClear}
                                    className="text-text-muted hover:text-danger-400"
                                >
                                    <X className="w-5 h-5" />
                                </Button>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Error message */}
            {error && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 flex items-center gap-2 text-sm text-danger-400 bg-danger-400/10 rounded-lg px-4 py-3"
                >
                    <FileWarning className="w-5 h-5 flex-shrink-0" />
                    {error}
                </motion.div>
            )}
        </div>
    );
}
