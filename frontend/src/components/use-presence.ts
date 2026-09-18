"use client";
import { useEffect, useState } from "react";

// Keep closing content mounted while CSS completes its exit transition.
export function usePresence<T>(value: T | null) {
  const [retained, setRetained] = useState(value);
  useEffect(() => {
    if (value !== null) { setRetained(value); return; }
    const delay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 220;
    const timer = window.setTimeout(() => setRetained(null), delay);
    return () => window.clearTimeout(timer);
  }, [value]);
  return value ?? retained;
}
