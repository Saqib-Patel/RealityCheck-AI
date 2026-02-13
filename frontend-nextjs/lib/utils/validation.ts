const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
const ALLOWED_VIDEO_TYPES = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime'];

export interface ValidationResult {
    isValid: boolean;
    error?: string;
}

export function validateFileSize(file: File): ValidationResult {
    if (file.size > MAX_FILE_SIZE) {
        return { isValid: false, error: `File too big (${(file.size / 1024 / 1024).toFixed(1)}MB). Max 5MB.` };
    }
    return { isValid: true };
}

export function validateFileType(file: File): ValidationResult {
    if (!ALLOWED_IMAGE_TYPES.includes(file.type) && !ALLOWED_VIDEO_TYPES.includes(file.type)) {
        return { isValid: false, error: `"${file.type}" not supported. Use JPG, PNG, WebP, MP4, AVI, or MOV.` };
    }
    return { isValid: true };
}

export function validateFile(file: File): ValidationResult {
    const size = validateFileSize(file);
    if (!size.isValid) return size;
    return validateFileType(file);
}

export const isImageFile = (file: File) => ALLOWED_IMAGE_TYPES.includes(file.type);
export const isVideoFile = (file: File) => ALLOWED_VIDEO_TYPES.includes(file.type);
export const getFileExtension = (name: string) => name.split('.').pop()?.toLowerCase() || '';

export function getFileTypeLabel(file: File): 'image' | 'video' | 'unknown' {
    if (isImageFile(file)) return 'image';
    if (isVideoFile(file)) return 'video';
    return 'unknown';
}

export { MAX_FILE_SIZE, ALLOWED_IMAGE_TYPES, ALLOWED_VIDEO_TYPES };
