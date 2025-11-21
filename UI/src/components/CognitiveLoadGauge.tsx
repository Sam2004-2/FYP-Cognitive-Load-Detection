import React from 'react';
import { LoadLevel } from '../types';

interface CognitiveLoadGaugeProps {
  load: number; // 0-1
}

const getLoadLevel = (load: number): LoadLevel => {
  if (load < 0.4) return 'low';
  if (load < 0.7) return 'medium';
  return 'high';
};

const CognitiveLoadGauge: React.FC<CognitiveLoadGaugeProps> = ({ load }) => {
  const level = getLoadLevel(load);
  
  const colorClasses = {
    low: 'bg-load-low',
    medium: 'bg-load-medium',
    high: 'bg-load-high'
  };

  return (
    <div className="flex items-center space-x-3">
      <div className="text-sm font-medium text-gray-700">Cognitive Load:</div>
      <div className={`${colorClasses[level]} px-4 py-2 rounded-full text-white font-semibold transition-all duration-300`}>
        {level.charAt(0).toUpperCase() + level.slice(1)}
      </div>
      <div className="text-sm text-gray-600">
        ({Math.round(load * 100)}%)
      </div>
    </div>
  );
};

export default CognitiveLoadGauge;

