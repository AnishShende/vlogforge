import React, { useState, useEffect, useRef } from 'react';
import { Film, Sparkles, Sliders, Clock, Loader2, Home, LayoutDashboard, Settings, Sun, Moon, Play, Share2, Download, XCircle } from 'lucide-react';
import UploadPanel from './components/UploadPanel';
import ProcessingMonitor from './components/ProcessingMonitor';
import VideoPreview from './components/VideoPreview';
import * as tus from 'tus-js-client';

import Login from './components/auth/Login';
import Register from './components/auth/Register';
import Dashboard from './components/Dashboard';

const CustomSelect = ({ value, onChange, options }) => {
  // ... keeping CustomSelect as is ...
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
  const [currentScreen, setCurrentScreen] = useState('loading'); // 'loading', 'login', 'register', 'dashboard', 'editor'
  
  // Editor state
  const [step, setStep] = useState(1);
  const [files, setFiles] = useState([]);
  const [contextText, setContextText] = useState('');
  const [jobId, setJobId] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [targetDuration, setTargetDuration] = useState(600);
  const [vlogGenre, setVlogGenre] = useState('default');
  const [theme, setTheme] = useState('dark');
  const [qualityThreshold, setQualityThreshold] = useState(0.20); // Conservative default
  const [projectName, setProjectName] = useState('Vlog Preview');
  const [currentProjectId, setCurrentProjectId] = useState(null);

  // Check auth on load
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('vlogforge_token');
      if (!token) {
        setCurrentScreen('login');
        return;
      }
      try {
        const res = await fetch('/api/auth/me', {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (res.ok) {
          setCurrentScreen('dashboard');
        } else {
          localStorage.removeItem('vlogforge_token');
          setCurrentScreen('login');
        }
      } catch (err) {
        console.error("Auth check failed:", err);
        setCurrentScreen('login');
      }
    };
    checkAuth();
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('vlogforge_token');
    setCurrentScreen('login');
  };

  const handleNewProject = async () => {
    try {
      const token = localStorage.getItem('vlogforge_token');
      const res = await fetch('/api/projects/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ title: 'New Vlog Project' })
      });
      if (res.ok) {
        const data = await res.json();
        setCurrentProjectId(data.id);
        setProjectName(data.title);
        resetEditorState();
        setCurrentScreen('editor');
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleUpdateProjectName = async () => {
    if (!currentProjectId) return;
    try {
      const token = localStorage.getItem('vlogforge_token');
      await fetch(`/api/projects/${currentProjectId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ title: projectName })
      });
    } catch (err) {
      console.error("Failed to update project name", err);
    }
  };

  const handleOpenProject = (project) => {
    setCurrentProjectId(project.id);
    setProjectName(project.title);
    resetEditorState();

    if (project.settings) {
      setContextText(project.settings.context_text || '');
      setTargetDuration(project.settings.target_duration || 600);
      setVlogGenre(project.settings.vlog_genre || 'default');
      setQualityThreshold(project.settings.quality_threshold || 0.20);
    }

    if (project.video_files && project.video_files.length > 0) {
      setFiles(project.video_files.map(f => ({
        id: f.id,
        name: f.filename,
        size: f.size_bytes,
        _duration: f.duration || 0,
        type: 'video/mp4',
        uploadStatus: 'complete'
      })));
    }

    const activeStages = ['processing', 'ingesting', 'transcribing', 'analyzing', 'classifying', 'edl_generating', 'assembling'];
    if (activeStages.includes(project.status)) {
      setJobId(project.id);
      setStep(2); // Jump straight to ProcessingMonitor
    } else if (project.status === 'complete') {
      setJobId(project.id);
      setStep(3); // Jump straight to Timeline Editor
    }
    
    setCurrentScreen('editor');
  };

  const handleRemoveFile = async (file) => {
    if (file.id && currentProjectId) {
      try {
        const token = localStorage.getItem('vlogforge_token');
        await fetch(`/api/projects/${currentProjectId}/files/${file.id}`, {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${token}` }
        });
      } catch (err) {
        console.error('Error removing file:', err);
      }
    }
  };

  const resetEditorState = () => {
    setStep(1);
    setFiles([]);
    setContextText('');
    setJobId(null);
    setDownloadUrl(null);
    setTargetDuration(10.0);
    setVlogGenre('default');
    setQualityThreshold(0.20);
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    
    try {
      const token = localStorage.getItem('vlogforge_token');
      const sortedFiles = [...files].sort((a, b) => a.lastModified - b.lastModified);

      // 1. Upload files via Tus
      for (const file of sortedFiles) {
        // Skip files that were restored from the database and are already uploaded
        if (!(file instanceof File) || file.uploadStatus === 'complete') {
          continue;
        }

        await new Promise((resolve, reject) => {
          const upload = new tus.Upload(file, {
            endpoint: "/api/upload/tus",
            chunkSize: 10 * 1024 * 1024, // 10MB chunks to prevent Vite proxy OOM crashes
            retryDelays: [0, 3000, 5000, 10000, 20000],
            metadata: {
              filename: unescape(encodeURIComponent(file.name)),
              filetype: file.type,
              project_id: currentProjectId
            },
            headers: {
              Authorization: `Bearer ${token}`
            },
            onError: (error) => {
              console.error("Failed because: " + error);
              reject(error);
            },
            onProgress: (bytesUploaded, bytesTotal) => {
              const percentage = ((bytesUploaded / bytesTotal) * 100).toFixed(2);
              console.log(file.name, percentage + "%");
            },
            onSuccess: () => {
              console.log("Download %s from %s", upload.file.name, upload.url);
              resolve();
            }
          });
          upload.start();
        });
      }

      // 2. Start Processing Job
      const formData = new FormData();
      formData.append('project_id', currentProjectId);
      formData.append('context_text', contextText);
      formData.append('vlog_genre', vlogGenre);
      formData.append('target_duration', targetDuration);
      formData.append('quality_threshold', qualityThreshold);

      const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Processing start failed');
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

  if (currentScreen === 'loading') {
    return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', background: 'var(--bg-main)', color: 'var(--primary)' }}><Loader2 className="spinner" size={32} /></div>;
  }

  if (currentScreen === 'login') {
    return <Login onLogin={() => setCurrentScreen('dashboard')} onNavigateRegister={() => setCurrentScreen('register')} />;
  }

  if (currentScreen === 'register') {
    return <Register onRegister={() => setCurrentScreen('dashboard')} onNavigateLogin={() => setCurrentScreen('login')} />;
  }

  if (currentScreen === 'dashboard') {
    return <Dashboard onNewProject={handleNewProject} onOpenProject={handleOpenProject} onLogout={handleLogout} />;
  }

  // currentScreen === 'editor'
  return (
    <div className="dashboard-layout">
      <div className="dashboard-main">
        {/* Top Header */}
        <header className="dashboard-header" style={{ padding: '0 1.5rem', height: '64px', borderBottom: '1px solid var(--card-border)' }}>
          {/* Left: Logo/Icon and Back */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', width: '300px' }}>
            <button onClick={() => setCurrentScreen('dashboard')} className="btn btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.6rem 1.2rem', background: 'linear-gradient(135deg, var(--primary) 0%, #a855f7 100%)', border: 'none', color: 'white', fontWeight: 700, letterSpacing: '0.05em' }}>
              <Home size={18} /> Dashboard
            </button>
          </div>
          
          {/* Center: Title Pill (Editable) */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center' }}>
            <div style={{ 
              background: 'var(--tab-group-bg)', 
              border: '1px solid var(--card-border)',
              borderRadius: '99px',
              padding: '0.4rem 1rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              fontSize: '0.8rem',
              transition: 'border-color 0.2s ease'
            }}
            onFocus={(e) => e.currentTarget.style.borderColor = 'var(--primary)'}
            onBlur={(e) => e.currentTarget.style.borderColor = 'var(--card-border)'}
            >
              <input 
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                onBlur={handleUpdateProjectName}
                style={{ 
                  background: 'transparent', border: 'none', outline: 'none', 
                  color: 'var(--text-main)', fontWeight: 500, width: `${Math.max(4, projectName.length)}ch`,
                  fontFamily: 'inherit', fontSize: '0.8rem', textAlign: 'center', minWidth: '50px'
                }} 
              />
              <span style={{ color: 'var(--text-muted)' }}>|</span>
              <span style={{ color: 'var(--text-disabled)' }}>{step === 3 ? 'AI Assembled' : step === 2 ? 'Processing' : 'Setup'}</span>
            </div>
          </div>
          
          {/* Right: Actions */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', width: '300px', justifyContent: 'flex-end' }}>
            {step === 3 && (
              <button 
                className="btn-primary" 
                onClick={() => {
                  if (downloadUrl) {
                    window.open(downloadUrl, '_blank');
                  } else {
                    alert('Download URL not ready yet.');
                  }
                }}
                style={{ 
                  padding: '0.6rem 1.2rem', 
                  borderRadius: '6px', 
                  fontSize: '0.75rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.5rem',
                  background: 'linear-gradient(135deg, var(--primary) 0%, #a855f7 100%)',
                  boxShadow: '0 4px 15px rgba(139, 92, 246, 0.4)',
                  border: 'none',
                  color: 'white',
                  fontWeight: 700,
                  letterSpacing: '0.05em',
                  transition: 'all 0.2s ease',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-1px)'; e.currentTarget.style.boxShadow = '0 6px 20px rgba(139, 92, 246, 0.6)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 15px rgba(139, 92, 246, 0.4)'; }}
              >
                <Download size={14} /> EXPORT
              </button>
            )}
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
                      { value: 60, label: '1 Min (Short Reel)' },
                      { value: 180, label: '3 Mins (Compact)' },
                      { value: 300, label: '5 Mins (Standard)' },
                      { value: 600, label: '10 Mins (Extended - Default)' },
                      { value: 900, label: '15 Mins (Documentary)' },
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
              onRemoveFile={handleRemoveFile}
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
