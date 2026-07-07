import React from 'react';
import { Download, RefreshCw, FileJson, Video, ExternalLink } from 'lucide-react';

export default function DownloadPanel({ jobId, downloadUrl, onReset }) {
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

  return (
    <div className="download-layout fade-in">
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
