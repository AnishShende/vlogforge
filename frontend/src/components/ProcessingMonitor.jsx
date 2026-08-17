import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { CheckCircle2, Loader2, AlertCircle, XCircle, StopCircle } from 'lucide-react';

const STAGES = [
  { key: 'ingesting', label: 'Ingesting & Pre-processing', desc: 'Extracting audio stream and identifying scene transitions.' },
  { key: 'transcribing', label: 'Transcribing Audio', desc: 'Whisper local model running speech-to-text.' },
  { key: 'analyzing', label: 'AI Keyframe & Context Analysis', desc: 'Gemini Vision parsing scene keyframes.' },
  { key: 'classifying', label: 'Segment Classification', desc: 'Labeling scenes with AI.' },
  { key: 'edl_generating', label: 'Generating EDL', desc: 'Applying target duration and pacing rules.' },
  { key: 'assembling', label: 'FFmpeg Vlog Assembly', desc: 'Cutting, normalizing, and joining clips.' }
];

export default function ProcessingMonitor({ jobId, onComplete, onFailed, onReset, onCancel }) {
  const [currentStage, setCurrentStage] = useState('pending');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('Initializing job request...');
  const hasFailedRef = useRef(false);

  const handleWebSocketMessage = useCallback((data) => {
    setCurrentStage(data.stage);
    setProgress(data.progress);
    setMessage(data.message);

    if (data.stage === 'complete') {
      onComplete(data.download_url);
    } else if (data.stage === 'failed' && !hasFailedRef.current) {
      hasFailedRef.current = true;
      onFailed();
    } else if (data.stage === 'not_found' && !hasFailedRef.current) {
      // Server restarted and lost the job — treat as failed so user can reset
      hasFailedRef.current = true;
      onFailed();
    }
  }, [onComplete, onFailed]);

  const { status } = useWebSocket(jobId, handleWebSocketMessage);

  const getStageStatus = (stageKey) => {
    const stageOrder = ['pending', 'ingesting', 'transcribing', 'analyzing', 'classifying', 'edl_generating', 'assembling', 'complete'];
    const currentIndex = stageOrder.indexOf(currentStage);
    const stageIndex = stageOrder.indexOf(stageKey);

    if (currentStage === 'failed') {
      return stageIndex < currentIndex ? 'completed' : stageKey === currentStage ? 'failed' : 'waiting';
    }

    if (currentStage === 'complete' || stageIndex < currentIndex) {
      return 'completed';
    } else if (stageKey === currentStage) {
      return 'active';
    } else {
      return 'waiting';
    }
  };

  // Circular Progress calculations
  const radius = 65;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  const isRunning = currentStage !== 'complete' && currentStage !== 'failed' && currentStage !== 'cancelled';

  return (
    <div className="studio-main fade-in" style={{ height: '100%', width: '100%' }}>

      {/* Center Canvas: Large Progress Ring + Cancel */}
      <div className="studio-canvas" style={{ flexDirection: 'column', gap: '2rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem', maxWidth: '400px' }}>
          
          {/* Premium Circular Progress Animation */}
          <div style={{ position: 'relative', width: '220px', height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            
            {/* Dynamic Core Background Glow */}
            <div style={{ 
              position: 'absolute', inset: '10px', 
              background: 'radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 60%)', 
              borderRadius: '50%', 
              animation: isRunning ? 'pulseGlow 3s infinite alternate ease-in-out' : 'none' 
            }} />

            <svg width="220" height="220" viewBox="0 0 220 220" style={{ position: 'absolute', zIndex: 2 }}>
              <defs>
                <linearGradient id="premium-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="var(--primary)" />
                  <stop offset="50%" stopColor="#06B6D4" />
                  <stop offset="100%" stopColor="#6D28D9" />
                </linearGradient>
                
                {/* Extremely premium inner glow filter */}
                <filter id="premium-glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
                <filter id="intense-glow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="12" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {/* Outer Slow Orbiting Dashed Ring */}
              <circle
                cx="110" cy="110" r="104"
                fill="none" stroke="rgba(255, 255, 255, 0.04)"
                strokeWidth="1.5" strokeDasharray="4 6"
                style={{ animation: isRunning ? 'spin 20s linear infinite' : 'none', transformOrigin: '110px 110px' }}
              />

              {/* Inner Fast Orbiting Solid Accent Ring */}
              <circle
                cx="110" cy="110" r="72"
                fill="none" stroke="rgba(139, 92, 246, 0.4)"
                strokeWidth="1" strokeDasharray="120 40 40 40"
                style={{ animation: isRunning ? 'spinCcw 10s linear infinite' : 'none', transformOrigin: '110px 110px' }}
              />

              {/* Main Background Track */}
              <circle
                cx="110" cy="110" r={radius}
                fill="transparent"
                stroke="rgba(255, 255, 255, 0.03)"
                strokeWidth="8"
              />

              {/* Intense Glow Layer for the Progress */}
              <circle
                cx="110" cy="110" r={radius}
                fill="transparent"
                stroke="url(#premium-gradient)"
                strokeWidth="8"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                filter="url(#intense-glow)"
                style={{ 
                  transition: 'stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1)', 
                  opacity: 0.6
                }}
                transform="rotate(-90 110 110)"
              />

              {/* Crisp Progress Stroke */}
              <circle
                cx="110" cy="110" r={radius}
                fill="transparent"
                stroke="url(#premium-gradient)"
                strokeWidth="8"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                filter="url(#premium-glow)"
                style={{ 
                  transition: 'stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
                }}
                transform="rotate(-90 110 110)"
              />
            </svg>

            {/* Center Text with Gradient Fill & Glow */}
            <div style={{ position: 'relative', zIndex: 3, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{
                fontSize: '2.5rem',
                fontWeight: '800',
                fontFamily: 'Outfit, sans-serif',
                background: 'linear-gradient(135deg, #fff 0%, #e2e8f0 50%, #6D28D9 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                lineHeight: 1,
                letterSpacing: '-0.02em'
              }}>
                {progress}%
              </span>
            </div>
          </div>
          
          {/* Status text */}
          <div style={{ textAlign: 'center', height: '90px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.35rem', fontFamily: 'Outfit, sans-serif' }}>
              {currentStage === 'failed' ? 'Processing Failed' : message}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
              {currentStage === 'failed' ? 'Terminated with Error' : `Status: ${status === 'connected' ? 'Rendering' : 'Connecting'}`}
            </div>
          </div>

          {/* Cancel button — prominent */}
          {isRunning && (
            <button 
              onClick={onCancel}
              style={{ 
                background: 'rgba(239, 68, 68, 0.06)',
                border: '1.5px solid rgba(239, 68, 68, 0.3)',
                color: '#f87171',
                fontSize: '0.95rem',
                fontWeight: 600,
                padding: '0.75rem 2rem',
                borderRadius: 'var(--radius-md)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                cursor: 'pointer',
                fontFamily: 'inherit',
                transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
                width: '280px',
                justifyContent: 'center',
                flexShrink: 0
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.12)';
                e.currentTarget.style.borderColor = '#ef4444';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.06)';
                e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.3)';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              <StopCircle size={18} /> Cancel Rendering
            </button>
          )}
        </div>
      </div>

      {/* Right Sidebar: Render Stages Checklist */}
      <div className="studio-sidebar-right">
        <h3 style={{ margin: '0 0 1.25rem 0', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-main)' }}>Render Stages</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          {STAGES.map((stage) => {
            const stageStatus = getStageStatus(stage.key);
            return (
              <div key={stage.key} style={{ 
                background: stageStatus === 'active' ? 'rgba(139, 92, 246, 0.04)' : 'transparent',
                border: '1px solid',
                borderColor: stageStatus === 'active' ? 'var(--primary-glow)' : stageStatus === 'completed' ? 'rgba(16, 185, 129, 0.15)' : 'var(--card-border)',
                padding: '0.65rem 0.75rem',
                borderRadius: 'var(--radius-sm)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                fontSize: '0.75rem',
                transition: 'all 0.4s ease'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: stageStatus === 'active' ? 600 : 400 }}>
                  {stageStatus === 'completed' && <CheckCircle2 size={13} style={{ color: 'var(--success)' }} />}
                  {stageStatus === 'active' && <Loader2 size={13} className="spinner" style={{ color: 'var(--primary)' }} />}
                  {stageStatus === 'failed' && <AlertCircle size={13} style={{ color: 'var(--danger)' }} />}
                  {stageStatus === 'waiting' && <CheckCircle2 size={13} style={{ color: 'var(--text-disabled)' }} />}
                  <span style={{ color: stageStatus === 'active' || stageStatus === 'completed' ? 'var(--text-main)' : 'var(--text-muted)' }}>
                    {stage.label}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Overall progress bar */}
        <div style={{ marginTop: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>
            <span>Overall Progress</span>
            <span style={{ color: 'var(--secondary)', fontWeight: 600 }}>{progress}%</span>
          </div>
          <div className="progress-bar-container">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
}
