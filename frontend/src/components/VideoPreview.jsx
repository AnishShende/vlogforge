import React, { useState, useEffect, useRef } from 'react';
import { Play, FileText, Scissors, XCircle, Pencil, RotateCcw, Plus, Check, Trash2, Video, Film, AlertTriangle, Loader2, GripVertical, Shield, Activity, Sliders, Sparkles } from 'lucide-react';

/**
 * Matching logic: checks if an EDL entry and a transcript/EGT segment overlap.
 * Supports both legacy (video_file/start/end) and new (source_file/start_sec/end_sec) formats.
 */
const segmentsMatch = (edlItem, transcriptSegment) => {
  if (!edlItem || !transcriptSegment) return false;

  const edlFile = edlItem.source_file || edlItem.video_file;
  const segFile = transcriptSegment.source_file || transcriptSegment.video_file;
  if (edlFile !== segFile) return false;

  const edlStart = edlItem.start_sec;
  const edlEnd = edlItem.end_sec;
  const segStart = transcriptSegment.start_sec ?? transcriptSegment.start;
  const segEnd = transcriptSegment.end_sec ?? transcriptSegment.end;

  const startMax = Math.max(edlStart, segStart);
  const endMin = Math.min(edlEnd, segEnd);
  const overlap = endMin - startMax;
  return overlap > 0.5 || (Math.abs(edlStart - segStart) < 1.6 && overlap > 0);
};

/**
 * Quality flag icon mapping for human-readable display.
 */
const QUALITY_FLAG_ICONS = {
  low_audio: { icon: '🔇', label: 'Low Audio' },
  shaky: { icon: '📐', label: 'Shaky' },
  overexposed: { icon: '💡', label: 'Overexposed' },
  bad_take: { icon: '🗑️', label: 'Bad Take' },
  very_short: { icon: '⏱️', label: 'Very Short' },
  short: { icon: '⏱️', label: 'Short' },
  high_disfluency: { icon: '🗣️', label: 'High Disfluency' },
  moderate_disfluency: { icon: '🗣️', label: 'Moderate Disfluency' },
  only_disfluencies: { icon: '❌', label: 'Only Disfluencies' },
  bad_take_phrase: { icon: '✂️', label: 'Bad Take Phrase' },
  background_noise: { icon: '📢', label: 'Background Noise' },
  low_speech_density: { icon: '💬', label: 'Low Speech Density' },
};

export default function VideoPreview({ jobId, downloadUrl, onReset, onReEdit }) {
  const [edl, setEdl] = useState([]);
  const [savedEdl, setSavedEdl] = useState([]);
  const [transcript, setTranscript] = useState([]);
  const [contextDoc, setContextDoc] = useState('');
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef(null);
  const timelineTrackRef = useRef(null);

  const [rawFiles, setRawFiles] = useState([]);
  const [viewMode, setViewMode] = useState('assembled'); // 'assembled' or 'original'
  const [activeRawFile, setActiveRawFile] = useState(null);
  const [isReRendering, setIsReRendering] = useState(false);
  const [renderedTimestamp, setRenderedTimestamp] = useState(Date.now());
  const pendingSeekRef = useRef(null);

  const [selectedSegment, setSelectedSegment] = useState(null);
  const [warnings, setWarnings] = useState([]);

  // Quality Threshold Calibration
  const [localQualityThreshold, setLocalQualityThreshold] = useState(0.35);
  const [isReReasoning, setIsReReasoning] = useState(false);

  // Layout Resizing States & Handlers
  const [timelineHeight, setTimelineHeight] = useState(240);
  const [sidebarLeftWidth, setSidebarLeftWidth] = useState(340);
  const [sidebarRightWidth, setSidebarRightWidth] = useState(300);

  const startResizingTimeline = (mouseDownEvent) => {
    mouseDownEvent.preventDefault();
    const startY = mouseDownEvent.clientY;
    const startHeight = timelineHeight;
    const doDrag = (mouseMoveEvent) => {
      const deltaY = startY - mouseMoveEvent.clientY;
      const newHeight = Math.max(100, Math.min(startHeight + deltaY, window.innerHeight * 0.6));
      setTimelineHeight(newHeight);
    };
    const stopDrag = () => {
      window.removeEventListener('mousemove', doDrag);
      window.removeEventListener('mouseup', stopDrag);
    };
    window.addEventListener('mousemove', doDrag);
    window.addEventListener('mouseup', stopDrag);
  };

  const startResizingLeftSidebar = (mouseDownEvent) => {
    mouseDownEvent.preventDefault();
    const startX = mouseDownEvent.clientX;
    const startWidth = sidebarLeftWidth;
    const doDrag = (mouseMoveEvent) => {
      const deltaX = mouseMoveEvent.clientX - startX;
      const newWidth = Math.max(250, Math.min(startWidth + deltaX, window.innerWidth * 0.6));
      setSidebarLeftWidth(newWidth);
    };
    const stopDrag = () => {
      window.removeEventListener('mousemove', doDrag);
      window.removeEventListener('mouseup', stopDrag);
    };
    window.addEventListener('mousemove', doDrag);
    window.addEventListener('mouseup', stopDrag);
  };

  const startResizingRightSidebar = (mouseDownEvent) => {
    mouseDownEvent.preventDefault();
    const startX = mouseDownEvent.clientX;
    const startWidth = sidebarRightWidth;
    const doDrag = (mouseMoveEvent) => {
      const deltaX = startX - mouseMoveEvent.clientX;
      const newWidth = Math.max(250, Math.min(startWidth + deltaX, window.innerWidth * 0.6));
      setSidebarRightWidth(newWidth);
    };
    const stopDrag = () => {
      window.removeEventListener('mousemove', doDrag);
      window.removeEventListener('mouseup', stopDrag);
    };
    window.addEventListener('mousemove', doDrag);
    window.addEventListener('mouseup', stopDrag);
  };

  useEffect(() => {
    // Fetch EDL
    fetch(`/api/jobs/${jobId}/edl`)
      .then((res) => {
        if (!res.ok) throw new Error('EDL not available');
        return res.json();
      })
      .then((data) => {
        setEdl(data.edl || []);
        setSavedEdl(data.edl || []);
      })
      .catch((err) => console.error(err));

    // Fetch Transcript (EGT segments)
    fetch(`/api/jobs/${jobId}/transcript`)
      .then((res) => {
        if (!res.ok) throw new Error('Transcript not available');
        return res.json();
      })
      .then((data) => {
        setTranscript(data.transcript || []);
        setContextDoc(data.context_document || '');
      })
      .catch((err) => console.error(err));

    // Fetch Job details for raw video durations and filenames
    fetch(`/api/jobs/${jobId}`)
      .then((res) => {
        if (!res.ok) throw new Error('Job details not available');
        return res.json();
      })
      .then((data) => {
        setRawFiles(data.files || []);
        if (data.quality_threshold !== undefined) {
          setLocalQualityThreshold(data.quality_threshold);
        }
        if (data.warnings) {
          setWarnings(data.warnings);
        }
      })
      .catch((err) => console.error(err));
  }, [jobId]);

  // Calculate cumulative start time offsets in the final concatenated video
  let cumulativeOffset = 0;
  const edlWithOffsets = edl.map((item) => {
    const startInFinal = cumulativeOffset;
    const dur = item.end_sec - item.start_sec;
    cumulativeOffset += dur;
    
    const match = transcript.find((t) => segmentsMatch(item, t));
    const transcriptText = match ? (match.transcript || match.text || '') : '';
    
    return { ...item, startInFinal, duration: dur, transcriptText };
  });

  // Helper: get the file field from a segment (supports both formats)
  const getSegFile = (seg) => seg.source_file || seg.video_file || '';
  const getSegStart = (seg) => seg.start_sec ?? seg.start ?? 0;
  const getSegEnd = (seg) => seg.end_sec ?? seg.end ?? 0;
  const getSegText = (seg) => seg.transcript || seg.text || '';
  const getSegType = (seg) => seg.segment_type || seg.label || 'SPEECH';

  // Group transcript items by kept vs cut, and order the kept ones to match the EDL
  const orderedTranscript = (() => {
    const kept = [];
    const cut = [];

    edl.forEach((edlItem) => {
      const match = transcript.find((t) => segmentsMatch(edlItem, t));
      if (match) {
        kept.push(match);
      }
    });

    transcript.forEach((t) => {
      const isKept = edl.some((e) => segmentsMatch(e, t));
      if (!isKept) {
        cut.push(t);
      }
    });

    return [...kept, ...cut];
  })();

  const isSegmentKept = (segment) => {
    return edl.some((e) => segmentsMatch(e, segment));
  };

  const toggleSegment = (segment) => {
    const isKept = isSegmentKept(segment);
    const segFile = getSegFile(segment);
    const segStart = getSegStart(segment);
    
    const newEdl = [];
    orderedTranscript.forEach(t => {
      const tFile = getSegFile(t);
      const tStart = getSegStart(t);
      const isCurrent = tFile === segFile && Math.abs(tStart - segStart) < 0.1;
      const shouldKeep = isCurrent ? !isKept : edl.some(e => segmentsMatch(e, t));
      if (shouldKeep) {
        const match = edl.find(e => segmentsMatch(e, t));
        if (match) {
          newEdl.push(match);
        } else {
          // Build a new EDL entry from the transcript/EGT segment
          newEdl.push({
            clip_id: t.clip_id || '',
            source_file: getSegFile(t),
            video_file: getSegFile(t),  // backward compat
            start_sec: getSegStart(t),
            end_sec: getSegEnd(t),
            editorial_type: getSegType(t),
            type: getSegType(t),  // backward compat
            human_modified: true,
            modification_type: 'added',
          });
        }
      }
    });
    setEdl(newEdl);
  };

  const handleSeek = (timeInFinal) => {
    if (videoRef.current && timeInFinal !== null) {
      videoRef.current.currentTime = timeInFinal;
      videoRef.current.play().catch(() => {});
    }
  };

  const handleSeekRaw = (filename, startSec) => {
    if (activeRawFile !== filename) {
      pendingSeekRef.current = startSec;
      setActiveRawFile(filename);
    } else {
      if (videoRef.current) {
        videoRef.current.currentTime = startSec;
        videoRef.current.play().catch(() => {});
      }
    }
  };

  const handleTranscriptBubbleClick = (item) => {
    // Select this segment for quality inspection
    setSelectedSegment(item);
    
    const isKept = isSegmentKept(item);
    if (isKept) {
      if (viewMode === 'assembled') {
        const finalOffset = getTranscriptOffset(item);
        if (finalOffset !== null) {
          handleSeek(finalOffset);
        }
      } else {
        handleSeekRaw(getSegFile(item), getSegStart(item));
      }
    } else {
      setViewMode('original');
      handleSeekRaw(getSegFile(item), getSegStart(item));
    }
  };

  const getTranscriptOffset = (item) => {
    const match = edlWithOffsets.find((e) => segmentsMatch(e, item));
    return match ? match.startInFinal : null;
  };

  const formatTime = (secs) => {
    if (isNaN(secs) || secs === null) return '00:00.0';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 10);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms}`;
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      if (activeRawFile) {
        const fileObj = rawFiles.find(f => f.filename === activeRawFile);
        setDuration(fileObj ? fileObj.duration : videoRef.current.duration);

        if (pendingSeekRef.current !== null) {
          videoRef.current.currentTime = pendingSeekRef.current;
          pendingSeekRef.current = null;
          videoRef.current.play().catch(() => {});
        }
      } else {
        setDuration(videoRef.current.duration);
      }
    }
  };

  const handleTimelineClick = (e) => {
    if (timelineTrackRef.current && videoRef.current && cumulativeOffset > 0) {
      const rect = timelineTrackRef.current.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const width = rect.width;
      const percentage = clickX / width;
      const seekTime = Math.max(0, Math.min(percentage * cumulativeOffset, cumulativeOffset));
      videoRef.current.currentTime = seekTime;
    }
  };

  const handleRawTrackClick = (e, file) => {
    if (e.currentTarget) {
      const rect = e.currentTarget.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const width = rect.width;
      const percentage = clickX / width;
      const seekTime = Math.max(0, Math.min(percentage * file.duration, file.duration));
      handleSeekRaw(file.filename, seekTime);
    }
  };

  const handleReRender = async () => {
    setIsReRendering(true);
    try {
      const response = await fetch(`/api/jobs/${jobId}/re-render`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ edl: edl }),
      });

      if (!response.ok) throw new Error('Re-render pipeline request failed');
      const data = await response.json();

      setSavedEdl(edl);
      setRenderedTimestamp(Date.now());
      setViewMode('assembled');
      setActiveRawFile(null);

      if (videoRef.current) {
        videoRef.current.load();
      }
      setIsReRendering(false);
    } catch (err) {
      console.error(err);
      alert('Error updating EDL: ' + err.message);
      setIsReRendering(false);
    }
  };

  const handleReReason = async () => {
    setIsReReasoning(true);
    try {
      const response = await fetch(`/api/jobs/${jobId}/re-reason`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ quality_threshold: localQualityThreshold })
      });
      if (!response.ok) throw new Error('Re-reasoning request failed');
      // Re-reasoning is async and will push updates via websocket (which App.jsx handles?)
      // Actually, if App.jsx handles websockets, it might switch back to Step 2.
      // We should tell App.jsx to reset step or listen to websocket here.
      // For now, trigger re-render overlay.
      alert('Re-evaluating quality and AI reasoning... Check terminal or wait for update.');
      setIsReReasoning(false);
    } catch (err) {
      console.error(err);
      alert('Error requesting re-reasoning: ' + err.message);
      setIsReReasoning(false);
    }
  };

  const downloadTranscripts = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(transcript, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", `subtitles_${jobId.slice(0, 8)}.json`);
    dlAnchorElem.click();
  };

  const downloadEDL = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(edl, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", `edl_${jobId.slice(0, 8)}.json`);
    dlAnchorElem.click();
  };

  // Playhead position percentage
  const playheadPositionPercent = cumulativeOffset > 0 ? (currentTime / cumulativeOffset) * 100 : 0;

  // Quality score color
  const getQualityColor = (score) => {
    if (score >= 0.7) return 'var(--success)';
    if (score >= 0.4) return 'var(--warning, #f59e0b)';
    return 'var(--danger)';
  };

  // Segment type color
  const getTypeColor = (type) => {
    switch (type) {
      case 'INTRO': return 'rgba(139, 92, 246, 0.4)';
      case 'OUTRO': return 'rgba(6, 182, 212, 0.4)';
      case 'SPEECH': case 'HIGHLIGHT': return 'rgba(236, 72, 153, 0.4)';
      case 'B_ROLL': return 'rgba(16, 185, 129, 0.4)';
      case 'SILENCE': return 'rgba(100, 100, 100, 0.3)';
      default: return 'rgba(236, 72, 153, 0.4)';
    }
  };

  const getTypeBorderColor = (type) => {
    switch (type) {
      case 'INTRO': return 'var(--primary)';
      case 'OUTRO': return 'var(--secondary)';
      case 'SPEECH': case 'HIGHLIGHT': return 'var(--accent)';
      case 'B_ROLL': return 'var(--success)';
      case 'SILENCE': return 'var(--text-disabled)';
      default: return 'var(--accent)';
    }
  };

  return (
    <div className="studio-main fade-in" style={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* Left Sidebar: Settings / AI Tools */}
        <div className="studio-sidebar-left" style={{ width: `${sidebarLeftWidth}px`, flexShrink: 0, padding: '1rem', background: 'var(--sidebar-bg)', display: 'flex', flexDirection: 'column' }}>
          
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FileText size={16} style={{ color: 'var(--primary)' }} /> EDL SEQUENCE
            </h3>
            <button style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
              <div style={{ width: '16px', height: '2px', background: 'currentColor', marginBottom: '4px' }} />
              <div style={{ width: '16px', height: '2px', background: 'currentColor', marginBottom: '4px' }} />
              <div style={{ width: '16px', height: '2px', background: 'currentColor' }} />
            </button>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', paddingRight: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {warnings.length > 0 && (
              <div style={{ padding: '0.75rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--danger)', borderRadius: 'var(--radius-sm)', marginBottom: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--danger)', fontSize: '0.8rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                  <AlertTriangle size={14} /> BUDGET OVERRIDE
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-main)', lineHeight: 1.4 }}>
                  {warnings.map((w, i) => <div key={i} style={{ marginBottom: '0.25rem' }}>{w}</div>)}
                  <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    You can override these cuts by clicking the transcript bubbles to re-add segments.
                  </div>
                </div>
              </div>
            )}
            
            {orderedTranscript.map((item, idx) => {
              const isKept = isSegmentKept(item);
              const offset = getTranscriptOffset(item);
              const type = getSegType(item);
              const dur = getSegEnd(item) - getSegStart(item);
              
              return (
                <div 
                  key={idx}
                  style={{
                    background: isKept ? 'var(--card-bg)' : 'rgba(255,255,255,0.02)',
                    border: '1px solid',
                    borderColor: isKept ? 'var(--card-border)' : 'rgba(255,255,255,0.05)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '0.75rem',
                    transition: 'all 0.2s',
                    opacity: isKept ? 1 : 0.5,
                    borderStyle: isKept ? 'solid' : 'dashed',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = isKept ? 'var(--primary)' : 'rgba(255,255,255,0.2)';
                    e.currentTarget.style.opacity = isKept ? 1 : 0.8;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = isKept ? 'var(--card-border)' : 'rgba(255,255,255,0.05)';
                    e.currentTarget.style.opacity = isKept ? 1 : 0.5;
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ fontSize: '0.7rem', fontWeight: 700, color: isKept ? getTypeBorderColor(type) : 'var(--text-disabled)' }}>
                        {isKept ? (type || 'CLIP') : 'CUT'}
                      </span>
                      {!isKept && (
                        <span style={{ fontSize: '0.6rem', padding: '0.1rem 0.3rem', background: 'var(--danger)', color: '#fff', borderRadius: '3px', fontWeight: 800 }}>
                          REMOVED
                        </span>
                      )}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        {isKept && offset !== null ? `${formatTime(offset)} - ${formatTime(offset + dur)}` : `${formatTime(getSegStart(item))} - ${formatTime(getSegEnd(item))}`}
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleSegment(item);
                        }}
                        style={{
                          background: isKept ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                          color: isKept ? 'var(--danger)' : '#10B981',
                          border: 'none',
                          borderRadius: '3px',
                          padding: '0.2rem 0.4rem',
                          fontSize: '0.6rem',
                          fontWeight: 800,
                          cursor: 'pointer',
                        }}
                      >
                        {isKept ? 'REMOVE' : 'RESTORE'}
                      </button>
                    </div>
                  </div>
                  <div 
                    onClick={() => handleTranscriptBubbleClick(item)}
                    style={{ fontSize: '0.75rem', color: 'var(--text-main)', lineHeight: 1.4, display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden', textDecoration: isKept ? 'none' : 'line-through', cursor: 'pointer' }}
                  >
                    {getSegText(item) || "No transcript available."}
                  </div>
                </div>
              );
            })}
            {orderedTranscript.length === 0 && (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textAlign: 'center', padding: '1rem' }}>
                No Transcript data available.
              </div>
            )}
          </div>
        </div>

        {/* Left vertical resizer handle */}
        <div
          className="layout-divider-vertical-left"
          onMouseDown={startResizingLeftSidebar}
          style={{
            width: '4px',
            cursor: 'col-resize',
            background: 'var(--card-border)',
            transition: 'background 0.2s',
            zIndex: 10,
            alignSelf: 'stretch',
            position: 'relative'
          }}
          onMouseEnter={(e) => e.target.style.background = 'var(--primary)'}
          onMouseLeave={(e) => e.target.style.background = 'var(--card-border)'}
        />

        {/* Center Canvas: Visual Video Monitor */}
        <div className="studio-canvas" style={{ position: 'relative' }}>
          {!activeRawFile && (
            <div style={{
              position: 'absolute', top: '1rem', left: '1rem',
              color: 'var(--text-muted)', fontSize: '0.65rem', fontWeight: 800,
              letterSpacing: '0.05em', textTransform: 'uppercase',
              display: 'flex', alignItems: 'center', gap: '0.4rem', zIndex: 10
            }}>
              <Film size={12} style={{ color: 'var(--primary)' }} /> PREVIEW MONITOR
            </div>
          )}
          {activeRawFile && (
            <div style={{
              position: 'absolute', top: '1rem', left: '1rem',
              color: '#EF4444', fontSize: '0.72rem', fontWeight: 700,
              display: 'flex', alignItems: 'center', gap: '0.5rem', zIndex: 20
            }}>
              <span style={{ width: '6px', height: '6px', background: '#EF4444', borderRadius: '50%', display: 'inline-block' }} />
              <span>PREVIEWING RAW: {activeRawFile}</span>
              <button
                onClick={() => setActiveRawFile(null)}
                style={{
                  background: 'rgba(255, 255, 255, 0.2)', border: 'none',
                  borderRadius: '3px', color: '#fff', padding: '0.15rem 0.4rem',
                  marginLeft: '0.5rem', cursor: 'pointer', fontSize: '0.65rem', fontWeight: 700
                }}
              >
                Return to Vlog
              </button>
            </div>
          )}
          <div className="studio-canvas-container" style={{ position: 'relative', width: '100%', height: '100%' }}>
              <video
              key={activeRawFile ? `raw-${activeRawFile}` : 'assembled'}
              ref={videoRef}
              src={activeRawFile ? `/api/jobs/${jobId}/raw/${activeRawFile}` : `${downloadUrl}?t=${renderedTimestamp}`}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              controls
              autoPlay
            />

          </div>
        </div>

        {/* Right vertical resizer handle */}
        <div
          className="layout-divider-vertical-right"
          onMouseDown={startResizingRightSidebar}
          style={{
            width: '4px',
            cursor: 'col-resize',
            background: 'var(--card-border)',
            transition: 'background 0.2s',
            zIndex: 10,
            alignSelf: 'stretch',
            position: 'relative'
          }}
          onMouseEnter={(e) => e.target.style.background = 'var(--primary)'}
          onMouseLeave={(e) => e.target.style.background = 'var(--card-border)'}
        />

        {/* Right Sidebar: Properties, Quality Inspector & Export */}
        {/* Right Sidebar: Project Media */}
        <div className="studio-sidebar-right" style={{ width: `${sidebarRightWidth}px`, flexShrink: 0, padding: '1rem', background: 'var(--sidebar-bg)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <h3 style={{ margin: 0, fontSize: '0.85rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Film size={14} style={{ color: 'var(--primary)' }} /> PROJECT MEDIA
            </h3>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{rawFiles?.length || 1} items</span>
          </div>


          <div style={{ marginTop: '0.5rem' }}>
            <h3 style={{ margin: 0, fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Project Details</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', fontSize: '0.7rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-disabled)' }}>Output:</span><span style={{ fontWeight: 600 }}>vlogforge_edit_{jobId?.slice(0,8)}.mp4</span></div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-disabled)' }}>Final Length:</span><span style={{ color: 'var(--text-main)', fontWeight: 600 }}>{formatTime(cumulativeOffset)}</span></div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '1rem' }}>
              <button className="btn btn-secondary" onClick={downloadTranscripts} style={{ padding: '0.4rem', fontSize: '0.7rem', border: 'none', background: 'rgba(255,255,255,0.05)', borderRadius: 'var(--radius-md)' }}>Export EGT</button>
              <button className="btn btn-secondary" onClick={downloadEDL} style={{ padding: '0.4rem', fontSize: '0.7rem', border: 'none', background: 'rgba(255,255,255,0.05)', borderRadius: 'var(--radius-md)' }}>Export EDL</button>
            </div>
            
            <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--card-border)', paddingTop: '1.5rem' }}>
              <h3 style={{ margin: 0, fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '1rem' }}>Quality Calibration</h3>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Threshold</span>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--primary)' }}>{localQualityThreshold.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0" max="1" step="0.05"
                value={localQualityThreshold}
                onChange={(e) => setLocalQualityThreshold(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--primary)', height: '4px', background: '#333', outline: 'none', borderRadius: '2px', WebkitAppearance: 'none' }}
              />
              
              <button
                className="btn btn-primary"
                onClick={handleReReason}
                disabled={isReReasoning}
                style={{ width: '100%', padding: '0.6rem', fontSize: '0.75rem', marginTop: '1rem', display: 'flex', justifyContent: 'center', gap: '0.5rem', borderRadius: 'var(--radius-md)' }}
              >
                {isReReasoning ? 'RE-EVALUATING...' : 'RE-GENERATE'}
              </button>
              
              {onReEdit && (
                <button
                  className="btn btn-secondary"
                  onClick={onReEdit}
                  style={{ width: '100%', padding: '0.5rem', fontSize: '0.7rem', marginTop: '0.5rem', border: 'none', background: 'rgba(255,255,255,0.05)', borderRadius: 'var(--radius-md)' }}
                >
                  Edit Project Setup
                </button>
              )}
            </div>
          </div>
        </div>

      </div>

      {/* Floating Re-render Banner if changes exist */}
      {JSON.stringify(edl) !== JSON.stringify(savedEdl) && (
        <div className="re-render-banner">
          <span className="re-render-banner-text">Timeline has manual modifications.</span>
          <button className="re-render-banner-btn" onClick={handleReRender}>
            Re-render Vlog
          </button>
        </div>
      )}

      {/* Re-rendering Loader Overlay */}
      {isReRendering && (
        <div className="re-render-overlay">
          <div className="re-render-box">
            <Loader2 className="spinner" size={32} style={{ color: 'var(--primary)' }} />
            <div className="re-render-title">Re-rendering Vlog</div>
            <div className="re-render-desc">
              FFmpeg is cutting and joining video segments into a new composition. This will take just a few seconds...
            </div>
          </div>
        </div>
      )}

      {/* Horizontal resizer handle */}
      <div
        className="layout-divider-horizontal"
        onMouseDown={startResizingTimeline}
        style={{
          height: '4px',
          cursor: 'row-resize',
          background: 'var(--card-border)',
          transition: 'background 0.2s',
          zIndex: 10,
          width: '100%',
          flexShrink: 0
        }}
        onMouseEnter={(e) => e.target.style.background = 'var(--primary)'}
        onMouseLeave={(e) => e.target.style.background = 'var(--card-border)'}
      />

      {/* Bottom Panel: Timeline Track */}
      <div className="studio-bottom-panel" style={{ height: `${timelineHeight}px`, flexShrink: 0, overflow: 'hidden' }}>
        <div className="timeline-workspace">

          {/* New NLE Toolbar */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.5rem 1rem', background: 'var(--header-bg)', borderBottom: '1px solid var(--card-border)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-main)', fontSize: '0.8rem', fontWeight: 800, letterSpacing: '0.05em' }}>
                <Film size={14} style={{ color: 'var(--primary)' }} /> NLE TIMELINE
              </div>
              <div style={{ width: '1px', height: '14px', background: 'var(--card-border)' }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', color: 'var(--text-main)' }}>
                <button style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer' }}><span style={{ transform: 'rotate(180deg)', display: 'inline-block' }}><Play size={16} fill="currentColor" /></span></button>
                <button style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer', padding: '0.5rem' }}><Play size={24} fill="currentColor" /></button>
                <button style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer' }}><Play size={16} fill="currentColor" /></button>
              </div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, fontFamily: 'monospace' }}>
                {formatTime(currentTime)} / {formatTime(cumulativeOffset)}
              </div>
            </div>
            

          </div>

          <div style={{ display: 'flex', height: 'calc(100% - 45px)', background: 'var(--bg-color)', overflowY: 'auto' }}>
            {/* Left Track Labels */}
            <div style={{ width: '100px', flexShrink: 0, background: 'var(--sidebar-bg)', borderRight: '1px solid var(--card-border)', display: 'flex', flexDirection: 'column' }}>
              <div style={{ height: '30px', borderBottom: '1px solid var(--card-border)' }} /> {/* Ruler spacer */}
              
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <div style={{ height: '60px', display: 'flex', alignItems: 'center', paddingLeft: '0.5rem', fontSize: '0.75rem', fontWeight: 700, color: '#10B981' }}>Video 1</div>
              </div>
            </div>

            {/* Timeline Tracks Area */}
            <div style={{ flex: 1, position: 'relative', overflowX: 'auto', display: 'flex', flexDirection: 'column' }}>
              
              {/* Ruler */}
              <div style={{ height: '30px', borderBottom: '1px solid var(--card-border)', display: 'flex', position: 'relative', background: 'var(--bg-color)' }}>
                {Array.from({length: Math.ceil(cumulativeOffset || 20)}).map((_, i) => {
                  const step = Math.max(1, Math.ceil((cumulativeOffset || 20) / 15));
                  if (i % step !== 0) return null;
                  return (
                    <div key={i} style={{ position: 'absolute', left: `${(i / (cumulativeOffset || 20)) * 100}%`, top: '50%', transform: 'translateX(-50%)', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                      {i}s
                    </div>
                  );
                })}
              </div>

              {/* Playhead */}
              <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${playheadPositionPercent}%`, width: '1px', background: '#fff', zIndex: 10 }}>
                <div style={{ position: 'absolute', top: '15px', left: '-4px', width: '9px', height: '9px', borderRadius: '50%', background: '#fff' }} />
              </div>

              <div style={{ flex: 1, position: 'relative' }} ref={timelineTrackRef} onClick={handleTimelineClick}>
                
                {/* Video 1 Track (Actual EDL) */}
                <div style={{ position: 'absolute', top: 0, height: '60px', width: '100%', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  {edlWithOffsets.map((item, idx) => {
                    const widthPercent = (item.duration / (cumulativeOffset || 20)) * 100;
                    const leftPercent = (item.startInFinal / (cumulativeOffset || 20)) * 100;
                    const isBlockActive = currentTime >= item.startInFinal && currentTime < item.startInFinal + item.duration;
                    return (
                      <div key={idx} style={{
                        position: 'absolute',
                        left: `${leftPercent}%`,
                        width: `${widthPercent}%`,
                        height: '40px',
                        top: '10px',
                        background: '#1F2937',
                        borderRadius: '4px',
                        border: isBlockActive ? '2px solid var(--primary)' : '1px solid #374151',
                        display: 'flex',
                        overflow: 'hidden',
                        cursor: 'pointer'
                      }} onClick={(e) => { e.stopPropagation(); handleSeek(item.startInFinal); }}>
                        <div style={{ padding: '0.2rem', fontSize: '0.65rem', color: '#fff', fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {item.editorial_type || 'Clip'}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Dummy tracks removed for simplicity */}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Re-rendering Loader Overlay */}
      {isReRendering && (
        <div className="re-render-overlay" style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div className="re-render-box" style={{ background: 'var(--card-bg)', padding: '2rem', borderRadius: 'var(--radius-lg)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', border: '1px solid var(--card-border)' }}>
            <Loader2 className="spinner" size={32} style={{ color: 'var(--primary)' }} />
            <div className="re-render-title" style={{ color: 'var(--text-main)', fontWeight: 700 }}>Re-rendering Vlog</div>
            <div className="re-render-desc" style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              FFmpeg is cutting and joining video segments into a new composition. This will take just a few seconds...
            </div>
          </div>
        </div>
      )}

      {/* Mock Export Modals (Controlled by state or just hidden for now, showing the structure) */}
      {/* 
      <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.9)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ background: 'var(--card-bg)', border: '1px solid var(--card-border)', padding: '2rem', borderRadius: 'var(--radius-lg)', display: 'flex', gap: '2rem', maxWidth: '600px' }}>
          <div style={{ flex: 1 }}>
            <div style={{ aspectRatio: '16/9', background: '#333', borderRadius: 'var(--radius-md)' }} />
          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.75rem' }}><span>Resolution</span> <span style={{color: 'white'}}>1080p</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.75rem' }}><span>Bit rate</span> <span style={{color: 'white'}}>High</span></div>
            <div style={{ width: '100%', height: '4px', background: '#333', borderRadius: '2px', marginTop: 'auto' }}><div style={{ width: '50%', height: '100%', background: 'var(--primary)' }} /></div>
            <button className="btn-secondary" style={{ width: '100%' }}>CANCEL</button>
          </div>
        </div>
      </div>
      */}

    </div>
  );
}
