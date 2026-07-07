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
    <div className="studio-main fade-in" style={{ height: 'calc(100vh - 60px)', width: '100vw' }}>

      {/* Center Canvas: Large Progress Ring + Cancel */}
      <div className="studio-canvas" style={{ flexDirection: 'column', gap: '2rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem', maxWidth: '400px' }}>
          
          {/* SVG Circular Progress Ring */}
          <svg className="processing-ring-svg" width="180" height="180" viewBox="0 0 180 180">
            <circle
              cx="90" cy="90" r={radius}
              fill="transparent"
              stroke="var(--card-border)"
              strokeWidth="7"
            />
            <circle
              cx="90" cy="90" r={radius}
              fill="transparent"
              stroke="var(--primary)"
              strokeWidth="7"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              style={{ transition: 'stroke-dashoffset 0.3s ease', filter: 'drop-shadow(0 0 8px var(--primary-glow))' }}
              transform="rotate(-90 90 90)"
            />
            <text
              x="90" y="96"
              textAnchor="middle"
              fill="var(--text-main)"
              fontSize="1.75rem"
              fontWeight="700"
              fontFamily="Outfit, sans-serif"
            >
              {progress}%
            </text>
          </svg>
          
          {/* Status text */}
          <div style={{ textAlign: 'center' }}>
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
                width: '100%',
                justifyContent: 'center'
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
