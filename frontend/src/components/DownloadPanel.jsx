import React, { useState, useEffect } from 'react';
import { Download, RefreshCw, FileJson, Video, Copy, Check, Tag, Hash, Clock, Type, AlignLeft } from 'lucide-react';

export default function DownloadPanel({ jobId, downloadUrl, onReset }) {
  const [metadata, setMetadata] = useState(null);
  const [copiedField, setCopiedField] = useState(null);
  const [editableTitle, setEditableTitle] = useState('');
  const [editableDescription, setEditableDescription] = useState('');

  useEffect(() => {
    if (!jobId) return;
    fetch(`/api/jobs/${jobId}/metadata`)
      .then((res) => {
        if (res.ok) return res.json();
        return null;
      })
      .then((data) => {
        if (data && data.metadata) {
          setMetadata(data.metadata);
          setEditableTitle(data.metadata.title || '');
          setEditableDescription(data.metadata.description || '');
        }
      })
      .catch(() => {});
  }, [jobId]);

  const copyToClipboard = (text, field) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedField(field);
      setTimeout(() => setCopiedField(null), 2000);
    });
  };

  const triggerDownload = (url, filename) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const downloadJson = (endpoint, filename) => {
    fetch(endpoint)
      .then((res) => res.json())
      .then((data) => {
        const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
          JSON.stringify(data, null, 2)
        )}`;
        triggerDownload(jsonString, filename);
      })
      .catch((err) => alert('Failed to export JSON: ' + err.message));
  };

  const CopyButton = ({ field, text, style }) => (
    <button
      onClick={() => copyToClipboard(text, field)}
      style={{
        background: copiedField === field ? 'rgba(34, 197, 94, 0.15)' : 'rgba(139, 92, 246, 0.1)',
        border: `1px solid ${copiedField === field ? 'rgba(34, 197, 94, 0.3)' : 'rgba(139, 92, 246, 0.2)'}`,
        borderRadius: '6px',
        padding: '0.4rem 0.7rem',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '0.35rem',
        fontSize: '0.72rem',
        fontWeight: 600,
        color: copiedField === field ? '#22c55e' : 'var(--primary)',
        transition: 'all 0.2s ease',
        whiteSpace: 'nowrap',
        ...style,
      }}
    >
      {copiedField === field ? <Check size={12} /> : <Copy size={12} />}
      {copiedField === field ? 'Copied!' : 'Copy'}
    </button>
  );

  return (
    <div className="download-layout fade-in" style={{ gap: '1.5rem' }}>
      {/* Primary Download Box */}
      <div className="download-box">
        <div className="download-icon-glow">
          <Video size={36} />
        </div>
        <h3 className="download-title">Vlog Generation Complete!</h3>
        <p className="download-subtitle">
          Your video has been edited, audio normalized, and fade transitions applied. It is fully ready for YouTube.
        </p>
        
        <div className="download-actions">
          <a href={downloadUrl} download className="btn btn-primary" style={{ padding: '1rem 2.25rem', fontSize: '1.05rem' }}>
            <Download size={20} /> Download Final MP4
          </a>
          <button className="btn btn-secondary" onClick={onReset}>
            <RefreshCw size={18} /> New Project
          </button>
        </div>
      </div>

      {/* M5: YouTube Metadata Section */}
      {metadata && (
        <div style={{
          background: 'var(--card-bg)',
          border: '1px solid var(--card-border)',
          borderRadius: 'var(--radius-lg)',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{
                width: '32px', height: '32px', borderRadius: '8px',
                background: 'linear-gradient(135deg, #ff0000 0%, #cc0000 100%)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
                  <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/>
                </svg>
              </div>
              <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700, fontFamily: 'Outfit, sans-serif' }}>
                YouTube Metadata
              </h3>
            </div>
            <button
              className="btn btn-secondary"
              style={{ padding: '0.45rem 0.8rem', fontSize: '0.72rem' }}
              onClick={() => downloadJson(`/api/jobs/${jobId}/metadata`, `vlogforge_metadata_${jobId.slice(0, 8)}.json`)}
            >
              <FileJson size={14} /> Export All
            </button>
          </div>

          {/* Title */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <Type size={13} /> Title
              </label>
              <CopyButton field="title" text={editableTitle} />
            </div>
            <input
              type="text"
              value={editableTitle}
              onChange={(e) => setEditableTitle(e.target.value)}
              maxLength={100}
              style={{
                background: 'var(--bg-main)',
                border: '1px solid var(--card-border)',
                borderRadius: '8px',
                padding: '0.7rem 0.9rem',
                color: 'var(--text-main)',
                fontSize: '0.95rem',
                fontWeight: 600,
                fontFamily: 'inherit',
                outline: 'none',
                transition: 'border-color 0.2s ease',
              }}
              onFocus={(e) => e.target.style.borderColor = 'var(--primary)'}
              onBlur={(e) => e.target.style.borderColor = 'var(--card-border)'}
            />
            <span style={{ fontSize: '0.65rem', color: 'var(--text-disabled)', alignSelf: 'flex-end' }}>
              {editableTitle.length}/100 characters
            </span>
          </div>

          {/* Description */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <AlignLeft size={13} /> Description
              </label>
              <CopyButton field="description" text={editableDescription} />
            </div>
            <textarea
              value={editableDescription}
              onChange={(e) => setEditableDescription(e.target.value)}
              rows={8}
              style={{
                background: 'var(--bg-main)',
                border: '1px solid var(--card-border)',
                borderRadius: '8px',
                padding: '0.7rem 0.9rem',
                color: 'var(--text-main)',
                fontSize: '0.82rem',
                fontFamily: 'inherit',
                lineHeight: 1.6,
                resize: 'vertical',
                outline: 'none',
                transition: 'border-color 0.2s ease',
              }}
              onFocus={(e) => e.target.style.borderColor = 'var(--primary)'}
              onBlur={(e) => e.target.style.borderColor = 'var(--card-border)'}
            />
          </div>

          {/* Tags */}
          {metadata.tags && metadata.tags.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <Tag size={13} /> Tags ({metadata.tags.length})
                </label>
                <CopyButton field="tags" text={metadata.tags.join(', ')} />
              </div>
              <div style={{
                display: 'flex', flexWrap: 'wrap', gap: '0.4rem',
                background: 'var(--bg-main)',
                border: '1px solid var(--card-border)',
                borderRadius: '8px',
                padding: '0.7rem 0.9rem',
              }}>
                {metadata.tags.map((tag, i) => (
                  <span key={i} style={{
                    background: 'rgba(139, 92, 246, 0.1)',
                    border: '1px solid rgba(139, 92, 246, 0.2)',
                    borderRadius: '99px',
                    padding: '0.25rem 0.65rem',
                    fontSize: '0.72rem',
                    fontWeight: 500,
                    color: 'var(--primary)',
                    whiteSpace: 'nowrap',
                  }}>
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Chapters */}
          {metadata.chapters && metadata.chapters.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <Clock size={13} /> Chapters ({metadata.chapters.length})
                </label>
                <CopyButton
                  field="chapters"
                  text={metadata.chapters.map(ch => `${ch.time} ${ch.label}`).join('\n')}
                />
              </div>
              <div style={{
                background: 'var(--bg-main)',
                border: '1px solid var(--card-border)',
                borderRadius: '8px',
                padding: '0.5rem 0',
                display: 'flex',
                flexDirection: 'column',
              }}>
                {metadata.chapters.map((ch, i) => (
                  <div key={i} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    padding: '0.45rem 0.9rem',
                    borderBottom: i < metadata.chapters.length - 1 ? '1px solid var(--card-border)' : 'none',
                    transition: 'background 0.15s ease',
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(139, 92, 246, 0.04)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    <span style={{
                      fontFamily: 'monospace',
                      fontSize: '0.78rem',
                      fontWeight: 700,
                      color: 'var(--secondary)',
                      minWidth: '50px',
                    }}>
                      {ch.time}
                    </span>
                    <span style={{
                      fontSize: '0.82rem',
                      color: 'var(--text-main)',
                      fontWeight: 500,
                    }}>
                      {ch.label}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Secondary Downloads (EDL + Transcript) */}
      <div className="secondary-downloads">
        <div className="file-dl-card">
          <div className="file-dl-header">
            <FileJson size={20} className="file-icon" />
            <span>Edit Decision List (EDL)</span>
          </div>
          <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Export the frame-accurate cutting timestamps in standard JSON format for reference or manual correction.
          </p>
          <button 
            className="btn btn-secondary" 
            style={{ width: '100%', padding: '0.65rem' }} 
            onClick={() => downloadJson(`/api/jobs/${jobId}/edl`, `vlogforge_edl_${jobId.slice(0, 8)}.json`)}
          >
            Export EDL JSON
          </button>
        </div>

        <div className="file-dl-card">
          <div className="file-dl-header">
            <FileJson size={20} className="file-icon" />
            <span>AI Labeled Transcript</span>
          </div>
          <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Export the transcribed dialog segments combined with their scene category classifications and visual keyframe analysis.
          </p>
          <button 
            className="btn btn-secondary" 
            style={{ width: '100%', padding: '0.65rem' }} 
            onClick={() => downloadJson(`/api/jobs/${jobId}/transcript`, `vlogforge_transcript_${jobId.slice(0, 8)}.json`)}
          >
            Export Transcript JSON
          </button>
        </div>
      </div>
    </div>
  );
}
