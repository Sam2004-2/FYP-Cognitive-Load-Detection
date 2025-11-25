import React from 'react';
import { FrameFeatures, WindowFeatures } from '../types/features';

interface LiveFeaturePanelProps {
  frameFeatures: FrameFeatures | null;
  windowFeatures: WindowFeatures | null;
  blinkCount: number;
  bufferFill: number;
}

const LiveFeaturePanel: React.FC<LiveFeaturePanelProps> = ({
  frameFeatures,
  windowFeatures,
  blinkCount,
  bufferFill,
}) => {
  const formatNumber = (value: number | undefined | null, decimals: number = 2): string => {
    if (value === null || value === undefined || isNaN(value)) return '—';
    return value.toFixed(decimals);
  };

  const getEarStatus = (ear: number | undefined): { color: string; label: string } => {
    if (ear === undefined || isNaN(ear)) return { color: 'bg-gray-400', label: 'Unknown' };
    if (ear < 0.21) return { color: 'bg-red-500', label: 'Closed' };
    if (ear < 0.25) return { color: 'bg-yellow-500', label: 'Partial' };
    return { color: 'bg-green-500', label: 'Open' };
  };

  const earStatus = getEarStatus(frameFeatures?.ear_mean);

  return (
    <div className="bg-slate-800 text-white rounded-lg p-4 font-mono text-xs">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-slate-300">Live Features</h3>
        <div className={`w-2 h-2 rounded-full ${frameFeatures?.valid ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
      </div>

      {/* Frame Features */}
      <div className="mb-4">
        <div className="text-slate-400 text-[10px] uppercase tracking-wide mb-2">Per-Frame</div>
        <div className="grid grid-cols-2 gap-2">
          {/* EAR with visual indicator */}
          <div className="bg-slate-700 rounded p-2">
            <div className="text-slate-400 text-[10px]">EAR (Mean)</div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold">{formatNumber(frameFeatures?.ear_mean, 3)}</span>
              <span className={`${earStatus.color} text-[9px] px-1.5 py-0.5 rounded text-white`}>
                {earStatus.label}
              </span>
            </div>
          </div>

          {/* Brightness */}
          <div className="bg-slate-700 rounded p-2">
            <div className="text-slate-400 text-[10px]">Brightness</div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold">{formatNumber(frameFeatures?.brightness, 0)}</span>
              <div className="flex-1 h-1.5 bg-slate-600 rounded overflow-hidden">
                <div 
                  className="h-full bg-yellow-400 transition-all duration-150"
                  style={{ width: `${Math.min(100, (frameFeatures?.brightness || 0) / 2.55)}%` }}
                />
              </div>
            </div>
          </div>

          {/* Left EAR */}
          <div className="bg-slate-700 rounded p-2">
            <div className="text-slate-400 text-[10px]">EAR Left</div>
            <span className="text-sm">{formatNumber(frameFeatures?.ear_left, 3)}</span>
          </div>

          {/* Right EAR */}
          <div className="bg-slate-700 rounded p-2">
            <div className="text-slate-400 text-[10px]">EAR Right</div>
            <span className="text-sm">{formatNumber(frameFeatures?.ear_right, 3)}</span>
          </div>

          {/* Quality */}
          <div className="bg-slate-700 rounded p-2 col-span-2">
            <div className="text-slate-400 text-[10px]">Detection Quality</div>
            <div className="flex items-center gap-2">
              <span className="text-sm">{formatNumber((frameFeatures?.quality || 0) * 100, 0)}%</span>
              <div className="flex-1 h-1.5 bg-slate-600 rounded overflow-hidden">
                <div 
                  className="h-full bg-blue-400 transition-all duration-150"
                  style={{ width: `${(frameFeatures?.quality || 0) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Window Features */}
      <div className="mb-4">
        <div className="text-slate-400 text-[10px] uppercase tracking-wide mb-2">Window Stats</div>
        <div className="grid grid-cols-2 gap-2">
          {/* Blink Rate */}
          <div className="bg-slate-700 rounded p-2">
            <div className="text-slate-400 text-[10px]">Blink Rate</div>
            <div className="flex items-baseline gap-1">
              <span className="text-lg font-bold">{formatNumber(windowFeatures?.blink_rate, 1)}</span>
              <span className="text-slate-400 text-[10px]">/min</span>
            </div>
          </div>

          {/* Blink Count */}
          <div className="bg-slate-700 rounded p-2">
            <div className="text-slate-400 text-[10px]">Blinks (Session)</div>
            <span className="text-lg font-bold">{blinkCount}</span>
          </div>

          {/* PERCLOS */}
          <div className="bg-slate-700 rounded p-2">
            <div className="text-slate-400 text-[10px]">PERCLOS</div>
            <div className="flex items-center gap-2">
              <span className="text-sm">{formatNumber((windowFeatures?.perclos || 0) * 100, 1)}%</span>
              <div className="flex-1 h-1.5 bg-slate-600 rounded overflow-hidden">
                <div 
                  className={`h-full transition-all duration-150 ${
                    (windowFeatures?.perclos || 0) > 0.15 ? 'bg-red-400' : 'bg-green-400'
                  }`}
                  style={{ width: `${Math.min(100, (windowFeatures?.perclos || 0) * 100 * 2)}%` }}
                />
              </div>
            </div>
          </div>

          {/* EAR Std */}
          <div className="bg-slate-700 rounded p-2">
            <div className="text-slate-400 text-[10px]">EAR Std</div>
            <span className="text-sm">{formatNumber(windowFeatures?.ear_std, 4)}</span>
          </div>

          {/* Mean Blink Duration */}
          <div className="bg-slate-700 rounded p-2 col-span-2">
            <div className="text-slate-400 text-[10px]">Mean Blink Duration</div>
            <div className="flex items-baseline gap-1">
              <span className="text-sm">{formatNumber(windowFeatures?.mean_blink_duration, 0)}</span>
              <span className="text-slate-400 text-[10px]">ms</span>
            </div>
          </div>
        </div>
      </div>

      {/* Buffer Status */}
      <div>
        <div className="text-slate-400 text-[10px] uppercase tracking-wide mb-2">Buffer</div>
        <div className="bg-slate-700 rounded p-2">
          <div className="flex justify-between items-center mb-1">
            <span className="text-slate-400 text-[10px]">Fill Level</span>
            <span className="text-[10px]">{(bufferFill * 100).toFixed(0)}%</span>
          </div>
          <div className="h-2 bg-slate-600 rounded overflow-hidden">
            <div 
              className={`h-full transition-all duration-300 ${
                bufferFill >= 1 ? 'bg-green-400' : 'bg-blue-400'
              }`}
              style={{ width: `${bufferFill * 100}%` }}
            />
          </div>
          {bufferFill < 1 && (
            <div className="text-slate-500 text-[9px] mt-1">
              Collecting frames...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiveFeaturePanel;

