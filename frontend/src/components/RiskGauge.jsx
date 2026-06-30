import React, { useEffect, useState } from 'react';

const RiskGauge = ({ score = 0, category = 'MEDIUM' }) => {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedScore(score);
    }, 100);
    return () => clearTimeout(timer);
  }, [score]);

  // Gauge calculations
  const radius = 60;
  const circumference = Math.PI * radius; // Half circle
  const dashoffset = circumference - (animatedScore / 100) * circumference;

  let color = 'var(--warning)';
  if (category === 'LOW') color = 'var(--success)';
  if (category === 'HIGH') color = 'var(--danger)';

  return (
    <div className="flex flex-col items-center justify-center relative w-full pt-4">
      <svg width="200" height="110" viewBox="0 0 200 110" className="overflow-visible">
        {/* Background Arc */}
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="var(--border-glass)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Foreground Arc */}
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circumference * (80/60)} // Scale strokeDasharray to match path radius vs generic calculation
          strokeDashoffset={((circumference * (80/60)) * (100 - animatedScore)) / 100}
          style={{ transition: 'stroke-dashoffset 1.5s cubic-bezier(0.22, 1, 0.36, 1)' }}
        />
        {/* Needle or Center Value */}
        <text
          x="100"
          y="90"
          textAnchor="middle"
          fill="var(--text-primary)"
          className="font-mono text-4xl font-bold"
          style={{ filter: `drop-shadow(0 0 8px ${color}40)` }}
        >
          {animatedScore.toFixed(1)}
        </text>
      </svg>
      <div className="font-mono text-sm tracking-widest mt-2 uppercase" style={{ color }}>
        {category} RISK
      </div>
    </div>
  );
};

export default RiskGauge;
