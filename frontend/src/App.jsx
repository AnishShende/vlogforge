import React, { useState, useEffect, useRef } from 'react';
import { Film, Sparkles, Sliders, Clock, Loader2, Home, LayoutDashboard, Settings } from 'lucide-react';
import UploadPanel from './components/UploadPanel';
import ProcessingMonitor from './components/ProcessingMonitor';
import VideoPreview from './components/VideoPreview';

const CustomSelect = ({ value, onChange, options }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedOption = options.find(o => o.value === value) || options[3];

  return (
    <div className="custom-select-wrapper" ref={dropdownRef} style={{ position: 'relative', flex: 1 }}>
      <div 
        className={`input-field custom-select-trigger ${isOpen ? 'active' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        style={{ 
          display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
          padding: '0.75rem 1rem', cursor: 'pointer', minHeight: 'auto'
        }}
      >
        <span style={{ fontSize: '0.82rem' }}>{selectedOption?.label}</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.3s ease', color: 'var(--text-muted)' }}>
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </div>
      
      {isOpen && (
        <div className="custom-select-menu" style={{
          position: 'absolute',
          bottom: 'calc(100% + 0.5rem)',
          left: 0,
          right: 0,
          background: 'var(--card-bg)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          border: '1px solid var(--card-border)',
          borderRadius: 'var(--radius-md)',
          boxShadow: '0 -10px 25px rgba(0,0,0,0.3)',
          overflow: 'hidden',
          zIndex: 50,
          display: 'flex',
          flexDirection: 'column'
        }}>
          {options.map((opt) => (
            <div 
              key={opt.value}
              className="custom-select-option"
              onClick={() => {
                onChange(opt.value);
                setIsOpen(false);
              }}
              style={{
                padding: '0.75rem 1rem',
                cursor: 'pointer',
                fontSize: '0.82rem',
                color: value === opt.value ? 'var(--primary)' : 'var(--text-main)',
                background: value === opt.value ? 'rgba(139, 92, 246, 0.1)' : 'transparent',
                transition: 'background 0.2s ease, color 0.2s ease',
              }}
              onMouseEnter={(e) => {
                if (value !== opt.value) e.currentTarget.style.background = 'rgba(139, 92, 246, 0.05)';
              }}
              onMouseLeave={(e) => {
                if (value !== opt.value) e.currentTarget.style.background = 'transparent';
              }}
            >
              {opt.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default function App() {
  const [step, setStep] = useState(1);
  const [files, setFiles] = useState([]);
  const [contextText, setContextText] = useState('');
  const [jobId, setJobId] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [targetDuration, setTargetDuration] = useState(10.0);
  const [vlogGenre, setVlogGenre] = useState('default');
  const [theme, setTheme] = useState('dark');
  const [qualityThreshold, setQualityThreshold] = useState(0.20); // Conservative default

  // Load theme from localStorage and apply to document
  useEffect(() => {
    const savedTheme = localStorage.getItem('vlogforge-theme') || 'dark';
    setTheme(savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('vlogforge-theme', newTheme);
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    const formData = new FormData();
    
    // Sort files chronologically based on their last modification time
    // This ensures that footage from cameras is processed in the correct order,
    // fixing issues where alphabetical sorting misplaced intros/outros.
    const sortedFiles = [...files].sort((a, b) => a.lastModified - b.lastModified);

    sortedFiles.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('context_text', contextText);
    formData.append('vlog_genre', vlogGenre);
    formData.append('target_duration', targetDuration);
    formData.append('quality_threshold', qualityThreshold);

    try {
      const response = await fetch('/api/jobs', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Upload failed');
      }

      const data = await response.json();
      setJobId(data.job_id);
      setStep(2);
    } catch (err) {
      console.error(err);
      alert('Error initiating job: ' + err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleProcessingComplete = (url) => {
    setDownloadUrl(url);
    setStep(3);
  };

  const handleProcessingFailed = () => {
    console.error('AI processing failed. Check the logs in the console.');
  };

  const handleCancel = async () => {
    if (!jobId) return;
    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
    } catch (err) {
      console.error('Error cancelling job:', err);
    }
    setStep(1);
    setJobId(null);
  };

  const resetProject = () => {
    setStep(1);
    setFiles([]);
    setContextText('');
    setJobId(null);
    setDownloadUrl(null);
    setTargetDuration(10.0);
    setVlogGenre('default');
    setQualityThreshold(0.20);
  };

  const handleReEdit = () => {
    setStep(1);
    setJobId(null);
    setDownloadUrl(null);
  };

  return (
    <div className="dashboard-layout">
      {/* Static Sidebar */}
      <aside className="dashboard-sidebar">
        <div style={{ marginBottom: '2rem', color: 'var(--primary)' }}>
          <Film size={28} />
        </div>
        <div className="dashboard-sidebar-item active" title="Home">
          <Home size={22} />
        </div>
        <div className="dashboard-sidebar-item" title="Projects">
          <LayoutDashboard size={22} />
        </div>
        <div style={{ marginTop: 'auto' }}>
          <div className="dashboard-sidebar-item" title="Settings">
            <Settings size={22} />
          </div>
        </div>
      </aside>

      <div className="dashboard-main">
        {/* Top Header */}
        <header className="dashboard-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
             <h1 style={{ margin: 0, fontSize: '1.25rem', fontFamily: 'Outfit, sans-serif' }}>VlogForge Studio</h1>
             <div style={{ width: '1px', height: '24px', background: 'var(--card-border)' }} />
             
             {/* Step indicators */}
             <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', fontSize: '0.8rem', userSelect: 'none' }}>
              {[
                { num: 1, label: 'Setup' },
                { num: 2, label: 'Processing' },
                { num: 3, label: 'Studio' },
              ].map((s) => (
                <div key={s.num} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: step === s.num ? 'var(--primary)' : 'var(--text-disabled)', boxShadow: step === s.num ? '0 0 6px var(--primary)' : 'none', transition: 'all 0.4s ease' }} />
                  <span style={{ color: step === s.num ? 'var(--text-main)' : 'var(--text-muted)', fontWeight: step === s.num ? 600 : 400, transition: 'all 0.4s ease' }}>{s.num}. {s.label}</span>
                </div>
              ))}
            </div>
          </div>
        </header>

        <main className="dashboard-content">

      {/* ═══ STEP 1: SETUP — 2-column split ═══ */}
      {step === 1 && (
        <div className="setup-split fade-in">

          {/* LEFT HALF: Prompt (top) + Pacing (bottom) + Generate */}
          <div className="setup-panel-left">
            {/* Creative Direction — takes the higher stage */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
                  <Sparkles size={18} style={{ color: 'var(--primary)', filter: 'drop-shadow(0 0 4px var(--primary-glow))' }} />
                  <h2 style={{ 
                    margin: 0, fontSize: '1.4rem', fontFamily: 'Outfit, sans-serif', fontWeight: 700
                  }}>
                    Creative Direction
                  </h2>
                </div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', margin: 0, lineHeight: 1.45 }}>
                  Tell the AI editor what to keep, cut, and how to structure your vlog.
                </p>
              </div>

              {/* Large prompt textarea */}
              <textarea
                id="context-input"
                className="input-field"
                style={{ 
                  flex: 1, width: '100%', boxSizing: 'border-box',
                  resize: 'none'
                }}
                placeholder="e.g. This is a daily vlog about my trip. Show the travel clips first — especially the flight window view — then the room tour. Cut any duplicate intros. Keep it energetic with quick transitions..."
                value={contextText}
                onChange={(e) => setContextText(e.target.value)}
              />
            </div>

            {/* Pacing Settings — lower section */}
            <div style={{ borderTop: '1px solid var(--card-border)', paddingTop: '1.25rem', marginTop: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <Sliders size={15} style={{ color: 'var(--secondary)' }} />
                <h3 style={{ margin: 0, fontSize: '0.9rem', letterSpacing: '0.03em', textTransform: 'uppercase', color: 'var(--text-main)' }}>Pacing Settings</h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <label htmlFor="duration-select" style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.3rem', whiteSpace: 'nowrap', width: '120px' }}>
                    <Clock size={12} />
                    Target Duration
                  </label>
                  <CustomSelect
                    value={targetDuration}
                    onChange={setTargetDuration}
                    options={[
                      { value: 1, label: '1 Min (Short Reel)' },
                      { value: 3, label: '3 Mins (Compact)' },
                      { value: 5, label: '5 Mins (Standard)' },
                      { value: 10, label: '10 Mins (Extended - Default)' },
                      { value: 15, label: '15 Mins (Documentary)' },
                      { value: 0, label: 'Auto (30% raw length)' }
                    ]}
                  />
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <label htmlFor="genre-select" style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.3rem', whiteSpace: 'nowrap', width: '120px' }}>
                    <Sparkles size={12} />
                    Vlog Genre
                  </label>
                  <CustomSelect
                    value={vlogGenre}
                    onChange={setVlogGenre}
                    options={[
                      { value: 'default', label: 'General / Default Vlog' },
                      { value: 'gym', label: 'Gym / Fitness Vlog' },
                      { value: 'travel', label: 'Travel Vlog' },
                      { value: 'daily', label: 'Daily Vlog / Lifestyle' },
                      { value: 'makeup', label: 'Makeup / Tutorial' }
                    ]}
                  />
                </div>
              </div>
            </div>

            {/* Generate Vlog Button */}
            <div style={{ marginTop: '1.5rem' }}>
              <button 
                className="btn btn-primary" 
                onClick={handleSubmit} 
                disabled={isSubmitting || files.length === 0} 
                style={{ 
                  width: '100%', padding: '0.9rem', borderRadius: 'var(--radius-md)', 
                  fontSize: '1rem', fontWeight: 700,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.6rem',
                  letterSpacing: '0.02em'
                }}
              >
                {isSubmitting ? (
                  <><Loader2 size={18} className="spinner" /> Uploading...</>
                ) : (
                  <><Sparkles size={18} /> Generate Vlog</>
                )}
              </button>
              {files.length === 0 && (
                <p style={{ color: 'var(--text-disabled)', fontSize: '0.72rem', margin: '0.6rem 0 0', textAlign: 'center' }}>
                  Add footage in the right panel to enable generation.
                </p>
              )}
            </div>
          </div>

          {/* Divider */}
          <div className="setup-divider" />

          {/* RIGHT HALF: Additional settings if any */}
          <div className="setup-panel-right">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
              <Sliders size={18} style={{ color: 'var(--secondary)' }} />
              <h2 style={{ margin: 0, fontSize: '1.4rem', fontFamily: 'Outfit, sans-serif', fontWeight: 700 }}>Quality Calibration</h2>
            </div>
            
            <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: '1.5rem', lineHeight: 1.45 }}>
              Set how aggressively the AI drops low-quality takes (stumbles, silence). You can fine-tune this later in the Studio.
            </p>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label htmlFor="quality-slider-setup" style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  Filter Threshold
                </label>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--secondary)' }}>{qualityThreshold.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                id="quality-slider-setup"
                min="0" max="1" step="0.05"
                value={qualityThreshold}
                onChange={(e) => setQualityThreshold(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--primary)' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-disabled)' }}>
                <span>Keep More (Conservative)</span>
                <span>Drop More (Aggressive)</span>
              </div>
            </div>

            <UploadPanel 
              files={files} 
              setFiles={setFiles} 
              isSubmitting={isSubmitting}
            />
          </div>

        </div>
      )}

      {/* ═══ STEP 2: PROCESSING ═══ */}
      {step === 2 && (
        <ProcessingMonitor 
          jobId={jobId} 
          onComplete={handleProcessingComplete} 
          onFailed={handleProcessingFailed} 
          onReset={resetProject}
          onCancel={handleCancel}
        />
      )}

      {/* ═══ STEP 3: PREVIEW / EXPORT ═══ */}
      {step === 3 && (
        <VideoPreview 
          jobId={jobId} 
          downloadUrl={downloadUrl} 
          onReset={resetProject}
          onReEdit={handleReEdit}
        />
      )}
        </main>
      </div>
    </div>
  );
}
