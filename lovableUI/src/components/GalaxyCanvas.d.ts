import type { ForwardRefExoticComponent, RefAttributes } from 'react';

export type GalaxyCoordinates = [number, number, number];

export interface GalaxyNodeMeta {
  id?: string;
  title?: string;
  domain?: string;
  score?: number;
  coords?: GalaxyCoordinates;
  [key: string]: any;
}

export interface GalaxyCanvasHandle {
  teleportCamera(
    targetCoords: GalaxyCoordinates,
    sourceCoords?: GalaxyCoordinates,
    meta?: GalaxyNodeMeta | null
  ): boolean;
  reset(): void;
}

export interface GalaxyCanvasProps {
  className?: string;
  sourceLabel?: string;
  targetLabel?: string;
  result?: any;
  onNodeSelect?: (node: GalaxyNodeMeta) => void;
}

declare const GalaxyCanvas: ForwardRefExoticComponent<
  GalaxyCanvasProps & RefAttributes<GalaxyCanvasHandle>
>;

export default GalaxyCanvas;
