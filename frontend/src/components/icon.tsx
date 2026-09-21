import type { SVGProps } from "react";

const paths = {
  wave: "M2 6c3-4 7 4 10 0s7 4 10 0M2 12c3-4 7 4 10 0s7 4 10 0M2 18c3-4 7 4 10 0s7 4 10 0",
  overview: "M12 16v-4m0-4h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z",
  layers: "m12 3 10 5-10 5L2 8l10-5Zm-10 9 10 5 10-5M2 16l10 5 10-5",
  legend: "M9 5h12M9 12h12M9 19h12M3 5h.01M3 12h.01M3 19h.01",
  chart: "M3 3v18h18M7 14l4-5 4 3 6-8",
  search: "m21 21-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Z",
  filter: "M4 5h16M7 12h10M10 19h4",
  close: "m6 6 12 12M6 18 18 6",
  home: "m3 10 9-7 9 7v11h-6v-7H9v7H3V10Z",
  play: "m7 4 14 8-14 8V4Z",
  pause: "M8 4v16M16 4v16",
  back: "m12 5-7 7 7 7M5 12h15",
  pin: "M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0ZM15 10a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z",
  up: "m5 15 5-5 4 4 7-9M15 5h6v6",
  down: "m5 9 5 5 4-4 7 9M15 19h6v-6",
};
export type IconName = keyof typeof paths;
export default function Icon({ name, ...props }: SVGProps<SVGSVGElement> & { name: IconName }) {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}><path d={paths[name]} /></svg>;
}
