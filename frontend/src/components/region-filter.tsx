"use client";

import { useEffect, useRef, useState } from "react";
import { regionOrder, regionPresets } from "@/lib/camera";
import type { RegionKey } from "@/lib/types";
import Icon from "./icon";

export default function RegionFilter({ value, onChange, onOpen }: { value: RegionKey; onChange: (value: RegionKey) => void; onOpen: () => void }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const close = () => { setOpen(false); trigger.current?.focus(); };

  useEffect(() => {
    if (!open) return;
    root.current?.querySelector<HTMLButtonElement>('[aria-pressed="true"]')?.focus();
    const dismiss = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", dismiss);
    return () => document.removeEventListener("pointerdown", dismiss);
  }, [open]);

  return <div ref={root} className="search-scope" onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false);
  }} onKeyDown={event => {
    if (event.key === "Escape" && open) { event.preventDefault(); event.stopPropagation(); close(); }
  }}>
    <button ref={trigger} type="button" className="region-filter-trigger" aria-label={`Filter wilayah: ${regionPresets[value].label}`} aria-expanded={open} aria-controls="region-filter-options" onClick={() => {
      if (open) close();
      else { onOpen(); setOpen(true); }
    }}><Icon name="filter" /></button>
    {open && <div id="region-filter-options" className="region-filter-options" role="group" aria-label="Cakupan pencarian">
      {regionOrder.map(key => <button key={key} type="button" aria-pressed={value === key} className={value === key ? "is-selected" : ""} onClick={() => {
        onChange(key);
        close();
      }}>
        <span>{regionPresets[key].label}</span>
        <span aria-hidden="true">{value === key ? "✓" : ""}</span>
      </button>)}
    </div>}
  </div>;
}
