import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileVideo, X, FolderOpen, Info, Clock, HardDrive } from 'lucide-react';

// Utility: read video duration from a File blob
function getVideoDuration(file) {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      window.URL.revokeObjectURL(video.src);
      resolve(video.duration || 0);
    };
    video.onerror = () => {
      window.URL.revokeObjectURL(video.src);
      resolve(0);
    };
    video.src = URL.createObjectURL(file);
  });
}

export default function UploadPanel({ files, setFiles, isSubmitting }) {
  const [durations, setDurations] = useState({}); // { fileName_size: durationInSecs }

  // Recalculate durations if files exist but durations state is empty (e.g., when returning from Step 2)
  useEffect(() => {
    const missingFiles = files.filter(f => !durations[`${f.name}_${f.size}`]);
    if (missingFiles.length > 0) {
      (async () => {
        const newDurations = {};
        for (const file of missingFiles) {
          const key = `${file.name}_${file.size}`;
          newDurations[key] = await getVideoDuration(file);
        }
        setDurations((prev) => ({ ...prev, ...newDurations }));
      })();
    }
  }, [files, durations]);

  const onDrop = useCallback(async (acceptedFiles) => {
    const videos = acceptedFiles.filter((file) => {
      const ext = file.name.split('.').pop().toLowerCase();
      return ['mp4', 'mov', 'avi', 'mkv'].includes(ext);
    });

    if (videos.length === 0) {
      alert('Please upload valid video formats (.mp4, .mov, .avi, or .mkv).');
      return;
    }

    // Read durations for new files
    const newDurations = {};
    for (const file of videos) {
      const key = `${file.name}_${file.size}`;
      const dur = await getVideoDuration(file);
      newDurations[key] = dur;
    }

    setDurations((prev) => ({ ...prev, ...newDurations }));

    setFiles((prev) => {
      const filtered = videos.filter(
        (v) => !prev.some((p) => p.name === v.name && p.size === v.size)
      );
      return [...prev, ...filtered];
    });
  }, [setFiles]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.mov', '.avi', '.mkv'] }
  });

  const removeFile = (index) => {
    const file = files[index];
    const key = `${file.name}_${file.size}`;
    setDurations((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const formatSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const formatDuration = (secs) => {
    if (!secs || isNaN(secs)) return '--:--';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const getDuration = (file) => {
    const key = `${file.name}_${file.size}`;
    return durations[key] || 0;
  };

  // Totals
  const totalSize = files.reduce((sum, f) => sum + f.size, 0);
  const totalDuration = files.reduce((sum, f) => sum + getDuration(f), 0);

  return (
    <div className="setup-panel-right fade-in">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <FolderOpen size={18} style={{ color: 'var(--primary)' }} />
        <h3 style={{ margin: 0, fontSize: '1.15rem', letterSpacing: '0.03em', fontFamily: 'Outfit, sans-serif', color: 'var(--text-main)' }}>Footage Assets</h3>
      </div>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', margin: '0 0 1.25rem 0', lineHeight: 1.4 }}>
        Drop your raw footage. Each clip's duration and size will be analyzed.
      </p>

      {/* Dropzone */}
      <div {...getRootProps()} className="dropzone" style={{ padding: '2rem 1.5rem', borderRadius: 'var(--radius-md)' }}>
        <input {...getInputProps()} />
        <div className="dropzone-icon" style={{ width: '48px', height: '48px' }}>
          <UploadCloud size={22} />
        </div>
        {isDragActive ? (
          <p style={{ fontWeight: 600, color: 'var(--primary)', fontSize: '0.85rem', margin: 0 }}>Drop videos here...</p>
        ) : (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', margin: 0 }}>
            Drag videos here, or <span style={{ color: 'var(--primary)', fontWeight: 600 }}>browse files</span>
          </p>
        )}
      </div>

      {/* Summary bar + file list */}
      {files.length > 0 && (
        <>
          {/* Combined totals summary */}
          <div style={{
            marginTop: '1.25rem',
            background: 'rgba(139, 92, 246, 0.04)',
            border: '1px solid rgba(139, 92, 246, 0.12)',
            borderRadius: 'var(--radius-sm)',
            padding: '0.75rem 1rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            flexWrap: 'wrap'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <FileVideo size={14} style={{ color: 'var(--primary)' }} />
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-main)' }}>
                {files.length} clip{files.length > 1 ? 's' : ''}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.75rem' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', color: 'var(--text-muted)' }}>
                <Clock size={12} style={{ color: 'var(--secondary)' }} />
                <span style={{ color: 'var(--secondary)', fontWeight: 600 }}>{formatDuration(totalDuration)}</span>
                total
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', color: 'var(--text-muted)' }}>
                <HardDrive size={12} style={{ color: 'var(--text-muted)' }} />
                <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>{formatSize(totalSize)}</span>
              </span>
            </div>
          </div>

          {/* Individual file list with duration + size */}
          <div className="file-list" style={{ marginTop: '0.75rem', maxHeight: '300px' }}>
            {files.map((file, idx) => (
              <div key={idx} className="file-item" style={{ padding: '0.6rem 0.85rem', fontSize: '0.8rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flex: 1, minWidth: 0 }}>
                  <FileVideo size={14} style={{ color: 'var(--primary)', flexShrink: 0 }} />
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div className="file-name" style={{ fontSize: '0.8rem' }}>{file.name}</div>
                    <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.15rem' }}>
                      <span style={{ fontSize: '0.68rem', color: 'var(--secondary)', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
                        <Clock size={10} />
                        {formatDuration(getDuration(file))}
                      </span>
                      <span className="file-size" style={{ fontSize: '0.68rem' }}>{formatSize(file.size)}</span>
                    </div>
                  </div>
                </div>
                <button className="remove-file-btn" onClick={() => removeFile(idx)} title="Remove file" disabled={isSubmitting}>
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Hoverable Inspector */}
      <div className="inspector-hover-trigger" style={{ marginTop: 'auto', alignSelf: 'flex-end', position: 'relative' }}>
        <div style={{ 
          display: 'flex', alignItems: 'center', gap: '0.4rem', 
          fontSize: '0.7rem', color: 'var(--text-disabled)', cursor: 'default',
          padding: '0.5rem 0.65rem', borderRadius: 'var(--radius-sm)',
          border: '1px solid transparent',
          transition: 'all 0.3s ease'
        }}>
          <Info size={12} />
          <span>Session Info</span>
        </div>
        <div className="inspector-hover-card">
          <div className="inspector-hover-card-inner">
            <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', fontWeight: 700, color: 'var(--text-main)', marginBottom: '0.65rem', letterSpacing: '0.05em' }}>Inspector</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.7rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Encoder</span>
                <span style={{ color: 'var(--success)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--success)' }} />
                  NVENC H.264
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--card-border)', paddingTop: '0.4rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>ASR</span>
                <span style={{ color: 'var(--success)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--success)' }} />
                  Whisper CUDA
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--card-border)', paddingTop: '0.4rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>LLM</span>
                <span style={{ color: 'var(--secondary)', fontWeight: 600 }}>Gemini 2.5 Flash</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--card-border)', paddingTop: '0.4rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>Loudness</span>
                <span>-14 LUFS</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--card-border)', paddingTop: '0.4rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>Crossfade</span>
                <span>0.5s</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
