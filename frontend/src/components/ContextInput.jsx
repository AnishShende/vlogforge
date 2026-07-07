import React from 'react';
import { Sliders, Clock } from 'lucide-react';

export default function ContextInput({ 
  contextText, 
  setContextText, 
  targetDuration, 
  setTargetDuration
}) {
  return (
    <div className="sidebar-section fade-in">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
        <Sliders size={16} style={{ color: 'var(--secondary)' }} />
        <h3 style={{ margin: 0, fontSize: '1rem', letterSpacing: '0.03em', textTransform: 'uppercase', color: 'var(--text-main)' }}>Pacing Settings</h3>
      </div>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', margin: '0 0 1.25rem 0', lineHeight: 1.4 }}>
        Configure the target duration and pacing for your final vlog edit.
      </p>

      <div className="form-group" style={{ marginBottom: '1.5rem' }}>
        <label htmlFor="duration-select" style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Clock size={12} />
          Target Vlog Duration
        </label>
        <select
          id="duration-select"
          className="input-field"
          style={{ background: 'rgba(0, 0, 0, 0.4)', color: '#fff', border: '1px solid var(--card-border)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem', width: '100%' }}
          value={targetDuration}
          onChange={(e) => setTargetDuration(parseFloat(e.target.value))}
        >
          <option value="1">1 Min (Short Reel)</option>
          <option value="3">3 Mins (Compact)</option>
          <option value="5">5 Mins (Standard)</option>
          <option value="10">10 Mins (Extended - Default)</option>
          <option value="15">15 Mins (Documentary)</option>
          <option value="0">Auto (30% raw length)</option>
        </select>
      </div>

      {/* Additional info cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.5rem' }}>
        <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid var(--card-border)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.72rem' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.62rem', textTransform: 'uppercase', fontWeight: 600, marginBottom: '0.3rem' }}>How it works</div>
          <div style={{ color: 'var(--text-muted)', lineHeight: 1.45 }}>
            Your raw footage is transcribed, analyzed by AI for scene content and quality, then cut and assembled into a polished vlog matching your creative direction.
          </div>
        </div>
        <div style={{ background: 'rgba(139, 92, 246, 0.03)', border: '1px solid rgba(139, 92, 246, 0.1)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.72rem' }}>
          <div style={{ color: 'var(--primary)', fontSize: '0.62rem', textTransform: 'uppercase', fontWeight: 600, marginBottom: '0.3rem' }}>Tip</div>
          <div style={{ color: 'var(--text-muted)', lineHeight: 1.45 }}>
            Be specific in your prompt — mention key moments, preferred ordering, and what to cut. The more detail, the better the edit.
          </div>
        </div>
      </div>
    </div>
  );
}
