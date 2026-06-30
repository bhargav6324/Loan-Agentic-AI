import React from 'react';

const AgentPipeline = ({ agentsExecuted = 0, status, sharedState = {} }) => {
  const nodes = [
    { key: 'data_agent', label: 'Data', desc: 'Ingestion & KYC' },
    { key: 'risk_agent', label: 'Risk', desc: 'ML Assessment' },
    { key: 'policy_agent', label: 'Policy', desc: 'Compliance' },
    { key: 'fairness_agent', label: 'Fairness', desc: 'Bias Audit' }
  ];

  // Determine current active node based on how many executed, or if processing
  let activeIndex = -1;
  if (status === 'processing') {
    activeIndex = agentsExecuted; // if 0 agents executed, index 0 is active.
    if (activeIndex >= nodes.length) activeIndex = nodes.length - 1;
  }

  const isNodeDone = (index) => {
    if (status === 'approved' || status === 'rejected' || status === 'escalated') return true;
    return index < agentsExecuted;
  };

  return (
    <div className="w-full flex flex-col items-center p-6 bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-glass)]" style={{ position: 'relative', overflow: 'hidden' }}>
      {/* Background radial glow */}
      <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 50% 50%, rgba(0, 212, 255, 0.05), transparent 70%)', pointerEvents: 'none' }} />
      
      <div className="flex items-center justify-between w-full relative z-10" style={{ maxWidth: '800px' }}>
        {nodes.map((node, i) => {
          const done = isNodeDone(i);
          const active = i === activeIndex;
          
          return (
            <React.Fragment key={node.key}>
              <div className="flex flex-col items-center" style={{ width: '120px', zIndex: 2 }}>
                <div 
                  className={`flex items-center justify-center rounded-full mb-2 transition-all duration-500
                    ${done ? 'bg-[var(--success-bg)] border-[var(--success)]' : active ? 'pulse-active bg-[var(--accent-glow)]' : 'bg-transparent border-[var(--text-muted)]'}
                  `}
                  style={{ 
                    width: '48px', height: '48px', 
                    borderWidth: '2px', borderStyle: 'solid',
                    color: done ? 'var(--success)' : active ? 'var(--accent)' : 'var(--text-muted)',
                    boxShadow: done ? '0 0 15px rgba(16, 185, 129, 0.2)' : 'none'
                  }}
                >
                  {done ? '✓' : (i + 1)}
                </div>
                <div className={`font-mono text-sm uppercase tracking-widest ${done || active ? 'text-primary' : 'text-muted'}`}>
                  {node.label}
                </div>
                <div className="text-[10px] text-secondary mt-1 tracking-tight text-center">
                  {node.desc}
                </div>
              </div>

              {i < nodes.length - 1 && (
                <div className="flex-1 flex items-center mx-2" style={{ height: '2px', zIndex: 1 }}>
                  <div 
                    className="w-full h-full transition-all duration-700" 
                    style={{ 
                      background: isNodeDone(i + 1) || active ? 'var(--accent)' : 'var(--border-glass)',
                      boxShadow: isNodeDone(i + 1) || active ? '0 0 8px var(--accent-glow)' : 'none',
                      opacity: 0.5
                    }} 
                  />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};

export default AgentPipeline;
