import React, { useState } from 'react';
import { NASATLXScores } from '../types';

interface NasaTLXFormProps {
  onSubmit: (scores: NASATLXScores) => void;
}

const NasaTLXForm: React.FC<NasaTLXFormProps> = ({ onSubmit }) => {
  const [scores, setScores] = useState<NASATLXScores>({
    mentalDemand: 50,
    physicalDemand: 50,
    temporalDemand: 50,
    performance: 50,
    effort: 50,
    frustration: 50
  });

  const dimensions = [
    { key: 'mentalDemand', label: 'Mental Demand', description: 'How mentally demanding was the task?' },
    { key: 'physicalDemand', label: 'Physical Demand', description: 'How physically demanding was the task?' },
    { key: 'temporalDemand', label: 'Temporal Demand', description: 'How hurried or rushed was the pace?' },
    { key: 'performance', label: 'Performance', description: 'How successful were you in accomplishing the task?' },
    { key: 'effort', label: 'Effort', description: 'How hard did you have to work?' },
    { key: 'frustration', label: 'Frustration', description: 'How insecure, discouraged, or stressed were you?' }
  ];

  const handleChange = (key: keyof NASATLXScores, value: number) => {
    setScores({ ...scores, [key]: value });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(scores);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <h3 className="text-xl font-semibold text-gray-800 mb-4">NASA Task Load Index (TLX)</h3>
      
      {dimensions.map(({ key, label, description }) => (
        <div key={key} className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="font-medium text-gray-700">{label}</label>
            <span className="text-sm font-semibold text-blue-600">{scores[key as keyof NASATLXScores]}</span>
          </div>
          <p className="text-sm text-gray-600">{description}</p>
          <input
            type="range"
            min="0"
            max="100"
            value={scores[key as keyof NASATLXScores]}
            onChange={(e) => handleChange(key as keyof NASATLXScores, parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <div className="flex justify-between text-xs text-gray-500">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>
      ))}

      <button
        type="submit"
        className="w-full bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors duration-200"
      >
        Submit Assessment
      </button>
    </form>
  );
};

export default NasaTLXForm;

