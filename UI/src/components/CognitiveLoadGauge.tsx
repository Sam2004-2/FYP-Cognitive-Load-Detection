import React from 'react';
import { LoadLevel } from '../types';

// Props interface defines the component's input - load is normalised 0-1 ***
interface CognitiveLoadGaugeProps {
  load: number; // 0-1
}

// Maps continuous load value to discrete levels for visual display ***
// Thresholds based on cognitive load research literature ***
const getLoadLevel = (load: number): LoadLevel => {
  if (load < 0.4) return 'low';
  if (load < 0.7) return 'medium';
  return 'high';
};

// React.FC is a TypeScript generic type for functional components ***
// Destructures 'load' directly from props for cleaner code ***
const CognitiveLoadGauge: React.FC<CognitiveLoadGaugeProps> = ({ load }) => {
  const level = getLoadLevel(load);
  
  // Lookup object maps level to Tailwind CSS classes defined in tailwind.config.js ***
  const colourClasses = {
    low: 'bg-load-low',
    medium: 'bg-load-medium',
    high: 'bg-load-high'
  };

  return (
    <div className="flex items-center space-x-3">
      <div className="text-sm font-medium text-gray-700">Cognitive Load:</div>
      <div className={`${colourClasses[level]} px-4 py-2 rounded-full text-white font-semibold transition-all duration-300`}>
        {level.charAt(0).toUpperCase() + level.slice(1)}
      </div>
      <div className="text-sm text-gray-600">
        ({Math.round(load * 100)}%)
      </div>
    </div>
  );
};

export default CognitiveLoadGauge;

