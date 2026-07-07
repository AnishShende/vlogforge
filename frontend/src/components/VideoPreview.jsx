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
      })
      .catch((err) => console.error(err));
  }, [jobId]);

  // Calculate cumulative start time offsets in the final concatenated video
  let cumulativeOffset = 0;
  const edlWithOffsets = edl.map((item) => {
    const startInFinal = cumulativeOffset;
    const dur = item.end_sec - item.start_sec;
    cumulativeOffset += dur;
    return { ...item, startInFinal, duration: dur };
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
    <div className="studio-main fade-in" style={{ height: 'calc(100vh - 60px)', width: '100vw', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* Left Sidebar: Transcripts Inspector */}
        <div className="studio-sidebar-left" style={{ width: `${sidebarLeftWidth}px`, flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <FileText size={16} style={{ color: 'var(--primary)' }} />
            <h3 style={{ margin: 0, fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-main)' }}>EGT Segments</h3>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', margin: '0 0 1.25rem 0', lineHeight: 1.4 }}>
            Click any segment to inspect quality. Include/exclude segments to modify the EDL.
          </p>

          <div className="transcript-flow" style={{ flex: 1, overflowY: 'auto', paddingRight: '0.25rem' }}>
            {orderedTranscript.map((item, idx) => {
              const isKept = isSegmentKept(item);
              const qualityScore = item.quality_score ?? 1.0;
              const isBadTake = qualityScore < localQualityThreshold;
              const segType = getSegType(item);
              const isSelected = selectedSegment && (item.clip_id === selectedSegment.clip_id ||
                (getSegFile(item) === getSegFile(selectedSegment) && Math.abs(getSegStart(item) - getSegStart(selectedSegment)) < 0.1));

              return (
                <div
                  key={idx}
                  className={`transcript-bubble ${isKept ? 'highlighted' : ''}`}
                  onClick={() => handleTranscriptBubbleClick(item)}
                  style={{
                    cursor: 'pointer',
                    opacity: isBadTake && !isKept ? 0.4 : isKept ? 1.0 : 0.5,
                    padding: '0.65rem',
                    fontSize: '0.8rem',
                    borderLeftWidth: '2px',
                    marginBottom: '0.65rem',
                    borderLeftColor: isBadTake ? 'var(--danger)' : undefined,
                    borderLeftStyle: isBadTake ? 'dashed' : undefined,
                    outline: isSelected ? '1px solid var(--primary)' : undefined,
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div className="bubble-meta" style={{ gap: '0.4rem', marginBottom: '0.2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <span className="bubble-label" style={{
                        fontSize: '0.65rem',
                        fontWeight: 700,
                        color: segType === 'INTRO' ? 'var(--primary)' :
                               segType === 'OUTRO' ? 'var(--secondary)' :
                               segType === 'SPEECH' || segType === 'HIGHLIGHT' ? 'var(--accent)' :
                               segType === 'B_ROLL' ? 'var(--success)' :
                               'var(--text-disabled)'
                      }}>{segType}</span>
                      <span className="bubble-time" style={{ fontSize: '0.7rem' }}>{formatTime(getSegStart(item))}</span>
                      {item.clip_id && (
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-disabled)', fontFamily: 'monospace' }}>
                          #{item.clip_id?.slice(0, 6)}
                        </span>
                      )}
                      {/* Quality score badge */}
                      <span style={{
                        fontSize: '0.6rem',
                        padding: '0.05rem 0.3rem',
                        borderRadius: '3px',
                        background: `${getQualityColor(qualityScore)}22`,
                        color: getQualityColor(qualityScore),
                        fontWeight: 600
                      }}>
                        {Math.round(qualityScore * 100)}%
                      </span>
                      {isBadTake && (
                        <span style={{ fontSize: '0.6rem', padding: '0.05rem 0.25rem', borderRadius: '3px', background: 'rgba(239,68,68,0.1)', color: 'var(--danger)', fontWeight: 600 }}>BAD TAKE</span>
                      )}
                      {!isKept && !isBadTake && (
                        <span style={{ fontSize: '0.6rem', padding: '0.05rem 0.25rem', borderRadius: '3px', background: 'rgba(239,68,68,0.1)', color: 'var(--danger)', fontWeight: 600 }}>CUT</span>
                      )}
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleSegment(item);
                      }}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: isKept ? 'var(--text-muted)' : 'var(--primary)',
                        cursor: 'pointer',
                        padding: '0.1rem',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        borderRadius: '3px'
                      }}
                      title={isKept ? "Exclude this segment" : "Include this segment"}
                    >
                      {isKept ? <XCircle size={13} /> : <Plus size={13} />}
                    </button>
                  </div>
                  <div style={{ lineHeight: 1.35 }}>{getSegText(item) || <i>(Visual scene / B-roll)</i>}</div>

                  {/* Quality flags inline */}
                  {item.quality_flags && item.quality_flags.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', marginTop: '0.3rem' }}>
                      {item.quality_flags.map((flag, flagIdx) => {
                        const flagInfo = QUALITY_FLAG_ICONS[flag] || { icon: '⚠️', label: flag };
                        return (
                          <span key={flagIdx} style={{
                            fontSize: '0.6rem',
                            padding: '0.05rem 0.3rem',
                            borderRadius: '3px',
                            background: 'rgba(239, 68, 68, 0.08)',
                            color: 'var(--text-muted)',
                          }} title={flagInfo.label}>
                            {flagInfo.icon} {flagInfo.label}
                          </span>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
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
        <div className="studio-canvas">
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
            {activeRawFile && (
              <div style={{
                position: 'absolute', top: '1.25rem', left: '1.25rem',
                background: 'rgba(239, 68, 68, 0.9)', backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                color: '#fff', fontSize: '0.72rem', fontWeight: 700,
                padding: '0.4rem 0.8rem', borderRadius: '4px',
                display: 'flex', alignItems: 'center', gap: '0.5rem', zIndex: 20,
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <span style={{ width: '6px', height: '6px', background: '#fff', borderRadius: '50%', display: 'inline-block' }} />
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
        <div className="studio-sidebar-right" style={{ width: `${sidebarRightWidth}px`, flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <Play size={16} style={{ color: 'var(--secondary)' }} />
            <h3 style={{ margin: 0, fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-main)' }}>Properties</h3>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', margin: '0 0 1.25rem 0', lineHeight: 1.4 }}>
            File details, quality inspection, and export actions.
          </p>

          {/* Video metadata */}
          <div style={{ background: 'var(--tab-group-bg)', border: '1px solid var(--card-border)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Output:</span><span style={{ fontWeight: 600 }}>vlogforge_edit_{jobId.slice(0,8)}.mp4</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Resolution:</span><span>1080p (30 FPS)</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>Final Length:</span><span style={{ color: 'var(--secondary)', fontWeight: 600 }}>{formatTime(cumulativeOffset)}</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>EDL Entries:</span><span>{edl.length}</span></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-muted)' }}>EGT Segments:</span><span>{transcript.length}</span></div>
          </div>

          {/* Global Quality Calibration */}
          <div style={{
            marginTop: '1rem',
            background: 'var(--tab-group-bg)',
            border: '1px solid var(--card-border)',
            borderRadius: 'var(--radius-sm)',
            padding: '0.75rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Sliders size={14} style={{ color: 'var(--primary)' }} />
              <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-main)' }}>AI Quality Threshold</span>
            </div>
            
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                <label htmlFor="quality-slider-studio" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  Filter Threshold
                </label>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--secondary)' }}>{localQualityThreshold.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                id="quality-slider-studio"
                min="0" max="1" step="0.05"
                value={localQualityThreshold}
                onChange={(e) => setLocalQualityThreshold(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--primary)' }}
                title="Higher threshold = more aggressive AI trimming"
              />
            </div>
            
            <button
              className="btn-primary"
              onClick={handleReReason}
              disabled={isReReasoning}
              style={{
                width: '100%',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.6rem',
                fontSize: '0.75rem'
              }}
            >
              {isReReasoning ? (
                <><Loader2 size={14} className="spin" /> Re-evaluating...</>
              ) : (
                <><Sparkles size={14} /> Re-Evaluate Quality</>
              )}
            </button>
          </div>

          {/* Quality Inspector Panel */}
          {selectedSegment && (
            <div style={{
              marginTop: '1rem',
              background: 'var(--tab-group-bg)',
              border: '1px solid var(--card-border)',
              borderRadius: 'var(--radius-sm)',
              padding: '0.75rem',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem' }}>
                <Activity size={14} style={{ color: 'var(--accent)' }} />
                <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-main)' }}>Quality Inspector</span>
              </div>

              {/* Clip ID */}
              {selectedSegment.clip_id && (
                <div style={{ fontSize: '0.7rem', color: 'var(--text-disabled)', fontFamily: 'monospace', marginBottom: '0.5rem' }}>
                  clip_id: {selectedSegment.clip_id}
                </div>
              )}

              {/* Quality score bar */}
              <div style={{ marginBottom: '0.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', marginBottom: '0.2rem' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Quality Score</span>
                  <span style={{ fontWeight: 700, color: getQualityColor(selectedSegment.quality_score ?? 1.0) }}>
                    {Math.round((selectedSegment.quality_score ?? 1.0) * 100)}%
                  </span>
                </div>
                <div style={{ height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.05)', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${(selectedSegment.quality_score ?? 1.0) * 100}%`,
                    background: getQualityColor(selectedSegment.quality_score ?? 1.0),
                    borderRadius: '3px',
                    transition: 'width 0.3s ease',
                  }} />
                </div>
              </div>

              {/* Segment info */}
              <div style={{ fontSize: '0.7rem', display: 'flex', flexDirection: 'column', gap: '0.3rem', marginBottom: '0.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Type:</span>
                  <span style={{ fontWeight: 600 }}>{getSegType(selectedSegment)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Duration:</span>
                  <span>{formatTime(getSegEnd(selectedSegment) - getSegStart(selectedSegment))}s</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Source:</span>
                  <span style={{ fontSize: '0.65rem' }}>{getSegFile(selectedSegment)}</span>
                </div>
                {selectedSegment.language_id && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Language:</span>
                    <span>{selectedSegment.language_id}</span>
                  </div>
                )}
              </div>

              {/* Quality flags detail */}
              {selectedSegment.quality_flags && selectedSegment.quality_flags.length > 0 && (
                <div style={{ borderTop: '1px solid var(--card-border)', paddingTop: '0.5rem' }}>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: '0.3rem', fontWeight: 600, textTransform: 'uppercase' }}>Quality Flags</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    {selectedSegment.quality_flags.map((flag, flagIdx) => {
                      const flagInfo = QUALITY_FLAG_ICONS[flag] || { icon: '⚠️', label: flag };
                      return (
                        <div key={flagIdx} style={{
                          fontSize: '0.7rem',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.4rem',
                          padding: '0.2rem 0.4rem',
                          borderRadius: '3px',
                          background: 'rgba(239, 68, 68, 0.06)',
                        }}>
                          <span>{flagInfo.icon}</span>
                          <span style={{ color: 'var(--text-main)' }}>{flagInfo.label}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Tags */}
              {selectedSegment.tags && selectedSegment.tags.length > 0 && (
                <div style={{ borderTop: '1px solid var(--card-border)', paddingTop: '0.5rem', marginTop: '0.5rem' }}>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: '0.3rem', fontWeight: 600, textTransform: 'uppercase' }}>Tags</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                    {selectedSegment.tags.map((tag, tagIdx) => (
                      <span key={tagIdx} style={{
                        fontSize: '0.6rem',
                        padding: '0.1rem 0.35rem',
                        borderRadius: '3px',
                        background: 'rgba(139, 92, 246, 0.1)',
                        color: 'var(--primary)',
                      }}>
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Override button */}
              <div style={{ marginTop: '0.75rem' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => toggleSegment(selectedSegment)}
                  style={{
                    width: '100%',
                    padding: '0.5rem',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.75rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.4rem',
                    background: 'var(--tab-group-bg)',
                  }}
                >
                  {isSegmentKept(selectedSegment)
                    ? <><XCircle size={13} /> Exclude from EDL</>
                    : <><Plus size={13} /> Include in EDL</>
                  }
                </button>
              </div>
            </div>
          )}

          {/* Download actions */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
            <a
              href={downloadUrl}
              download
              className="btn btn-primary"
              style={{ padding: '0.65rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
            >
              Download Video
            </a>

            <button
              className="btn btn-secondary"
              onClick={downloadTranscripts}
              style={{ padding: '0.65rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', background: 'var(--tab-group-bg)' }}
            >
              Export EGT JSON
            </button>

            <button
              className="btn btn-secondary"
              onClick={downloadEDL}
              style={{ padding: '0.65rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', background: 'var(--tab-group-bg)' }}
            >
              Export EDL JSON
            </button>
          </div>

          {/* Project actions */}
          <div style={{ borderTop: '1px solid var(--card-border)', paddingTop: '1.5rem', marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {onReEdit && (
              <button
                className="btn btn-secondary"
                onClick={onReEdit}
                style={{ width: '100%', padding: '0.65rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem' }}
              >
                <Pencil size={14} /> Edit Prompt & Re-generate
              </button>
            )}
            <button
              className="btn btn-secondary"
              onClick={onReset}
              style={{ width: '100%', padding: '0.65rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem', borderColor: 'var(--card-border)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem' }}
            >
              <RotateCcw size={14} /> Start New Project
            </button>
          </div>

          {/* AI Creative Insight card */}
          <div style={{
            marginTop: '1.25rem',
            background: 'rgba(139, 92, 246, 0.02)',
            border: '1px solid var(--card-border)',
            borderRadius: 'var(--radius-sm)',
            padding: '0.75rem'
          }}>
            <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--primary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>AI Creative Insight</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>{contextDoc || "No creative details logged."}</div>
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

          <div className="timeline-toolbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Scissors size={14} style={{ color: 'var(--accent)' }} />
                <span style={{ fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Timeline</span>
              </div>
              <div className="timeline-tab-group">
                <button
                  className={`timeline-tab ${viewMode === 'assembled' ? 'active' : ''}`}
                  onClick={() => {
                    setViewMode('assembled');
                    setActiveRawFile(null);
                  }}
                >
                  Assembled Vlog
                </button>
                <button
                  className={`timeline-tab ${viewMode === 'original' ? 'active' : ''}`}
                  onClick={() => setViewMode('original')}
                >
                  Original Footage
                </button>
              </div>
              {/* Phase 0 notice */}
              <div style={{
                display: 'flex', alignItems: 'center', gap: '0.3rem',
                fontSize: '0.65rem', color: 'var(--text-disabled)',
                padding: '0.2rem 0.5rem',
                borderRadius: '3px',
                background: 'rgba(139, 92, 246, 0.05)',
                border: '1px solid var(--card-border)',
              }}>
                <Shield size={10} />
                <span>Phase 0: Chronological order locked</span>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span>Playhead:</span>
              <span className="timeline-playhead-time">
                {formatTime(currentTime)}
              </span>
              <span>/</span>
              <span>
                {formatTime(activeRawFile ? (rawFiles.find(f => f.filename === activeRawFile)?.duration || 0) : cumulativeOffset)}
              </span>
            </div>
          </div>

          {viewMode === 'assembled' ? (
            <div className="timeline-track-container">
              <div className="timeline-track" ref={timelineTrackRef} onClick={handleTimelineClick}>

                {/* Playhead indicator line */}
                <div
                  className="timeline-playhead-line"
                  style={{ left: `${playheadPositionPercent}%` }}
                >
                  <div className="timeline-playhead-handle" />
                </div>

                {/* Render block segments */}
                {edlWithOffsets.map((item, idx) => {
                  const widthPercent = (item.duration / cumulativeOffset) * 100;
                  const segType = item.editorial_type || item.type || 'KEEP';
                  const typeColor = getTypeColor(segType);
                  const borderColor = getTypeBorderColor(segType);
                  const isBlockActive = currentTime >= item.startInFinal && currentTime < item.startInFinal + item.duration;

                  return (
                    <div
                      key={idx}
                      className={`timeline-block ${isBlockActive ? 'active' : ''}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleSeek(item.startInFinal);
                      }}
                      style={{
                        width: `${widthPercent}%`,
                        background: typeColor,
                        borderColor: borderColor,
                        borderLeft: '2px solid',
                        borderLeftColor: borderColor,
                        boxShadow: isBlockActive ? `0 0 8px ${borderColor}` : 'none',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                      }}
                      title={`${segType}: ${item.source_file || item.video_file} (${formatTime(item.duration)}s)`}
                    >
                      <div className="timeline-block-title" style={{ color: '#fff' }}>{segType}</div>
                      <div className="timeline-block-duration">{formatTime(item.duration)}s</div>
                    </div>
                  );
                })}

              </div>
            </div>
          ) : (
            <div className="timeline-multi-track-container">
              {rawFiles.map((file, fileIdx) => {
                const fileSegments = transcript.filter(s => getSegFile(s) === file.filename);

                // Track dynamic playhead for this file track
                let isTrackActive = false;
                let playheadPercent = 0;

                if (activeRawFile) {
                  if (activeRawFile === file.filename) {
                    isTrackActive = true;
                    if (file.duration > 0) {
                      playheadPercent = (currentTime / file.duration) * 100;
                    }
                  }
                } else {
                  const activeEdlItem = edlWithOffsets.find(
                    item => currentTime >= item.startInFinal && currentTime < item.startInFinal + item.duration
                  );
                  const activeFile = activeEdlItem ? (activeEdlItem.source_file || activeEdlItem.video_file) : null;
                  if (activeEdlItem && activeFile === file.filename) {
                    isTrackActive = true;
                    if (file.duration > 0) {
                      const currentSecInRaw = activeEdlItem.start_sec + (currentTime - activeEdlItem.startInFinal);
                      playheadPercent = (currentSecInRaw / file.duration) * 100;
                    }
                  }
                }

                return (
                  <div key={fileIdx} className="timeline-track-wrapper">
                    <div className="timeline-track-header">
                      <span className="timeline-track-title">{file.filename}</span>
                      <span className="timeline-track-duration">{formatTime(file.duration)}</span>
                    </div>

                    <div
                      className="timeline-track"
                      onClick={(e) => handleRawTrackClick(e, file)}
                    >
                      {/* Playhead indicator line */}
                      {isTrackActive && (
                        <div
                          className="timeline-playhead-line"
                          style={{ left: `${playheadPercent}%` }}
                        >
                          <div className="timeline-playhead-handle" />
                        </div>
                      )}

                      {/* Render segments for this file */}
                      {fileSegments.map((segment, segIdx) => {
                        const isKept = isSegmentKept(segment);
                        const qualityScore = segment.quality_score ?? 1.0;
                        const isBadTake = qualityScore < localQualityThreshold;
                        const segDur = getSegEnd(segment) - getSegStart(segment);
                        const widthPercent = (segDur / file.duration) * 100;
                        const segType = getSegType(segment);
                        const typeColor = getTypeColor(segType);
                        const borderColor = getTypeBorderColor(segType);

                        const isBlockActive = isTrackActive && (
                          activeRawFile
                            ? (currentTime >= getSegStart(segment) && currentTime < getSegEnd(segment))
                            : (() => {
                                const activeEdlItem = edlWithOffsets.find(
                                  item => currentTime >= item.startInFinal && currentTime < item.startInFinal + item.duration
                                );
                                return activeEdlItem && segmentsMatch(activeEdlItem, segment);
                              })()
                        );

                        return (
                          <div
                            key={segIdx}
                            className={`timeline-block ${isBlockActive ? 'active' : ''} ${isKept ? '' : 'cut'}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSeekRaw(file.filename, getSegStart(segment));
                              setSelectedSegment(segment);
                            }}
                            style={{
                              width: `${widthPercent}%`,
                              background: isKept ? typeColor : isBadTake ? 'rgba(239, 68, 68, 0.1)' : undefined,
                              borderColor: isBadTake ? 'var(--danger)' : borderColor,
                              borderLeft: isBadTake ? '2px dashed' : '2px solid',
                              borderLeftColor: isBadTake ? 'var(--danger)' : borderColor,
                              boxShadow: isBlockActive && isKept ? `0 0 8px ${borderColor}` : 'none'
                            }}
                            title={`${segType}: ${formatTime(segDur)}s (${getSegText(segment) || 'Visual scene'})${isBadTake ? ' [BAD TAKE]' : ''}`}
                          >
                            <div className="timeline-block-title" style={{ color: isKept ? '#fff' : isBadTake ? 'var(--danger)' : 'var(--text-disabled)' }}>
                              {segType}
                            </div>
                            <div className="timeline-block-duration">
                              {formatTime(segDur)}s
                            </div>

                            <button
                              className="timeline-block-hover-action"
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleSegment(segment);
                              }}
                            >
                              {isKept ? <XCircle size={11} /> : <Plus size={11} />}
                              <span>{isKept ? 'Exclude' : 'Include'}</span>
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
