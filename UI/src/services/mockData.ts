import { CognitiveLoadData } from '../types';

let currentLoad = 0.5;
let trend = 0;

export const generateMockCognitiveLoad = (): number => {
  // Gradual, realistic changes to simulate actual cognitive load patterns
  const randomChange = (Math.random() - 0.5) * 0.1;
  
  // Add some trend to make it more realistic
  if (Math.random() > 0.8) {
    trend = (Math.random() - 0.5) * 0.2;
  }
  
  currentLoad += randomChange + trend * 0.1;
  
  // Keep within bounds
  currentLoad = Math.max(0.1, Math.min(0.9, currentLoad));
  
  // Dampen trend over time
  trend *= 0.9;
  
  return currentLoad;
};

export const generateMockSessionData = (durationMinutes: number): CognitiveLoadData[] => {
  const dataPoints: CognitiveLoadData[] = [];
  const intervalSeconds = 2; // Data point every 2 seconds
  const totalPoints = (durationMinutes * 60) / intervalSeconds;
  
  // Reset for consistent generation
  currentLoad = 0.3 + Math.random() * 0.2;
  trend = 0;
  
  for (let i = 0; i < totalPoints; i++) {
    dataPoints.push({
      timestamp: i * intervalSeconds,
      load: generateMockCognitiveLoad()
    });
  }
  
  return dataPoints;
};

export const resetMockData = () => {
  currentLoad = 0.5;
  trend = 0;
};

