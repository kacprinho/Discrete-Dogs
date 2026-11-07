import type { ForwardRefExoticComponent, RefAttributes } from 'react';
export type GalaxyCoordinates = [number, number, number];
export interface GalaxyCanvasHandle {
  teleportCamera(targetCoords: GalaxyCoordinates, sourceCoords?: GalaxyCoordinates): boolean;
  reset(): void;
}
export interface GalaxyCanvasProps {
  className?: string;
  sourceLabel?: string;
  targetLabel?: string;
}
declare const GalaxyCanvas: ForwardRefExoticComponent<GalaxyCanvasProps & RefAttributes<GalaxyCanvasHandle>>;
export default GalaxyCanvas;
