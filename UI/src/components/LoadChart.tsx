import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { CognitiveLoadData } from '../types';

interface LoadChartProps {
  data: CognitiveLoadData[];
}

const LoadChart: React.FC<LoadChartProps> = ({ data }) => {
  const chartData = data.map((point) => ({
    time: `${Math.floor(point.timestamp / 60)}:${String(point.timestamp % 60).padStart(2, '0')}`,
    load: Math.round(point.load * 100)
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="time" 
          label={{ value: 'Time (MM:SS)', position: 'insideBottom', offset: -5 }}
        />
        <YAxis 
          label={{ value: 'Cognitive Load (%)', angle: -90, position: 'insideLeft' }}
          domain={[0, 100]}
        />
        <Tooltip />
        <Line 
          type="monotone" 
          dataKey="load" 
          stroke="#3b82f6" 
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default LoadChart;

