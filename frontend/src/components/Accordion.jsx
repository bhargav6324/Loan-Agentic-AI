import React, { useState } from 'react';

const Accordion = ({ title, icon, defaultOpen = false, children, status = 'neutral' }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  let statusColor = 'var(--text-muted)';
  if (status === 'success') statusColor = 'var(--success)';
  if (status === 'warning') statusColor = 'var(--warning)';
  if (status === 'danger') statusColor = 'var(--danger)';

  return (
    <div className="mb-4 border border-[var(--border-glass)] rounded-md bg-[rgba(0,0,0,0.2)] overflow-hidden transition-all duration-300 hover:border-[rgba(0,212,255,0.3)]">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 bg-transparent border-none cursor-pointer text-left focus:outline-none"
      >
        <div className="flex items-center gap-3">
          <span className="text-xl" style={{ color: statusColor }}>{icon}</span>
          <span className="font-mono text-sm tracking-widest uppercase text-secondary">{title}</span>
        </div>
        <span className="text-muted text-lg transition-transform duration-300" style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>
          ▼
        </span>
      </button>
      
      <div 
        className="transition-all duration-300 ease-in-out" 
        style={{ 
          maxHeight: isOpen ? '5000px' : '0px', 
          opacity: isOpen ? 1 : 0,
          borderTop: isOpen ? '1px solid var(--border-glass)' : 'none'
        }}
      >
        <div className="p-4 text-sm text-secondary leading-relaxed font-sans">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Accordion;
