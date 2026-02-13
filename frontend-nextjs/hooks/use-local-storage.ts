'use client';

import { useState, useEffect } from 'react';

export function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T | ((prev: T) => T)) => void] {
    const [storedValue, setStoredValue] = useState<T>(initialValue);
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
        setIsClient(true);
        try {
            const item = window.localStorage.getItem(key);
            if (item) setStoredValue(JSON.parse(item));
        } catch (e) {
            console.warn(`localStorage read failed for "${key}":`, e);
        }
    }, [key]);

    const setValue = (value: T | ((prev: T) => T)) => {
        try {
            const next = value instanceof Function ? value(storedValue) : value;
            setStoredValue(next);
            if (isClient) window.localStorage.setItem(key, JSON.stringify(next));
        } catch (e) {
            console.warn(`localStorage write failed for "${key}":`, e);
        }
    };

    return [storedValue, setValue];
}
