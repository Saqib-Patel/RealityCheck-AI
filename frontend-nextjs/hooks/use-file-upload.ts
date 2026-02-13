'use client';

import { useState } from 'react';
import { validateFile, isImageFile, isVideoFile } from '@/lib/utils/validation';

const MAX_FILE_SIZE = 5 * 1024 * 1024;

export function useFileUpload() {
    const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
    const [errors, setErrors] = useState<string[]>([]);

    function addFiles(files: File[]) {
        const valid: File[] = [];
        const errs: string[] = [];

        files.forEach(file => {
            const result = validateFile(file);
            result.isValid ? valid.push(file) : errs.push(result.error!);
        });

        setErrors(errs);
        if (valid.length) setSelectedFiles(valid);
    }

    function removeFile(index: number) {
        setSelectedFiles(prev => prev.filter((_, i) => i !== index));
        setErrors([]);
    }

    function clearFiles() {
        setSelectedFiles([]);
        setErrors([]);
    }

    return {
        selectedFiles,
        errors,
        addFiles,
        removeFile,
        clearFiles,
        hasFiles: selectedFiles.length > 0,
        primaryFile: selectedFiles[0] || null,
    };
}

export { MAX_FILE_SIZE, isImageFile, isVideoFile };
