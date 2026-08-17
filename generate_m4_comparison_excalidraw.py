import json
import uuid

def gen_id():
    return str(uuid.uuid4())

elements = []

def add_element(el):
    elements.append(el)
    return el["id"]

def create_rect(x, y, w, h, bg_color, stroke_color, stroke_style="solid", opacity=100, roughness=0, roundness=3):
    return {
        "id": gen_id(),
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": stroke_color,
        "backgroundColor": bg_color,
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": stroke_style,
        "roughness": roughness,
        "opacity": opacity,
        "groupIds": [],
        "roundness": {"type": roundness},
        "seed": 1,
        "version": 1,
        "versionNonce": 1,
        "isDeleted": False,
        "boundElements": []
    }

def create_text(x, y, text, font_size=16, color="#000", font_family=1, align="center", bg_color="transparent"):
    lines = text.split('\n')
    width = len(max(lines, key=len)) * (font_size * 0.6)
    height = len(lines) * (font_size * 1.25)
    return {
        "id": gen_id(),
        "type": "text",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": color,
        "backgroundColor": bg_color,
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "seed": 1,
        "version": 1,
        "versionNonce": 1,
        "isDeleted": False,
        "boundElements": [],
        "text": text,
        "fontSize": font_size,
        "fontFamily": font_family,
        "textAlign": align,
        "verticalAlign": "middle",
        "baseline": font_size
    }

def create_arrow(start_x, start_y, end_x, end_y, waypoints=None, stroke_color="#9ca3af"):
    a_id = gen_id()
    points = [[0, 0]]
    if waypoints:
        for wp in waypoints:
            points.append([wp[0] - start_x, wp[1] - start_y])
    points.append([end_x - start_x, end_y - start_y])

    arrow_el = {
        "id": a_id,
        "type": "arrow",
        "x": start_x,
        "y": start_y,
        "width": abs(end_x - start_x),
        "height": abs(end_y - start_y),
        "angle": 0,
        "strokeColor": stroke_color,
        "backgroundColor": "transparent",
        "fillStyle": "hachure",
        "strokeWidth": 3,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "roundness": {"type": 2},
        "seed": 1,
        "version": 1,
        "versionNonce": 1,
        "isDeleted": False,
        "boundElements": [],
        "points": points,
        "lastCommittedPoint": None,
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "triangle"
    }
    add_element(arrow_el)
    return a_id

def create_node(x, y, title, description, bg_color="#f8fafc", stroke_color="#cbd5e1", title_color="#0f172a", w=440, h=160):
    group_id = gen_id()
    
    rect_el = create_rect(x, y, w, h, bg_color, stroke_color, roundness=3)
    rect_el["groupIds"] = [group_id]
    add_element(rect_el)
    
    title_el = create_text(x + 15, y + 15, title, font_size=18, color=title_color, font_family=2, align="left")
    title_el["groupIds"] = [group_id]
    add_element(title_el)
    
    desc_el = create_text(x + 15, y + 50, description, font_size=14, color="#4b5563", font_family=3, align="left")
    desc_el["groupIds"] = [group_id]
    add_element(desc_el)
    
    return {"id": rect_el["id"], "x": x, "y": y, "w": w, "h": h}

def connect(c1, c2, color="#9ca3af", offset_y=0):
    start_x = c1["x"] + c1["w"]//2
    start_y = c1["y"] + c1["h"]
    end_x = c2["x"] + c2["w"]//2
    end_y = c2["y"]
    create_arrow(start_x, start_y, end_x, end_y, stroke_color=color)

# ------------- LAYOUT -------------

# Title
add_element(create_text(500, 50, "VlogForge Architecture: Pre vs Post M4 (Long-Footage Scaling)", 32, "#0f172a", 2, "center"))
add_element(create_text(500, 100, "Max Capacity target updated to 3 Hours of raw footage.", 18, "#64748b", 3, "center"))

# Columns
PRE_X = 100
POST_X = 650

add_element(create_text(PRE_X + 220, 160, "CURRENT (PRE-M4)", 24, "#ef4444", 2, "center"))
add_element(create_text(POST_X + 220, 160, "PLANNED (POST-M4)", 24, "#22c55e", 2, "center"))

# Step 1: Ingest
pre_ingest = create_node(PRE_X, 220, "1. Ingest (tasks/ingest.py)", "Inputs: Raw MP4s\n- Parallel FFmpeg Transcode to CFR\n- Extract Audio to WAV\n- PySceneDetect (Cascade: Hard Cuts + Adaptive)\nOutput: Raw EGTSegments + Keyframes")
post_ingest = create_node(POST_X, 220, "1. Ingest (tasks/ingest.py)", "[UNCHANGED]\nInputs: Raw MP4s\n- Parallel FFmpeg Transcode to CFR\n- Extract Audio to WAV\n- PySceneDetect (Cascade)\nOutput: Raw EGTSegments + Keyframes")

# Step 2: Transcribe
pre_transcribe = create_node(PRE_X, 420, "2. Transcribe (tasks/transcribe.py)", "Inputs: Audio WAV\n- faster-whisper (local) or Gemini STT API\n- align_transcript_with_segments() (O(N+M) sweep)\nOutput: Segments with Transcript Text")
post_transcribe = create_node(POST_X, 420, "2. Transcribe (tasks/transcribe.py)", "[UNCHANGED]\nInputs: Audio WAV\n- faster-whisper (local) or Gemini STT API\n- align_transcript_with_segments()\nOutput: Segments with Transcript Text")

# Step 3: Subdivide
pre_subdivide = create_node(PRE_X, 620, "3. Subdivide (tasks/scene_detect.py)", "- subdivide_by_speech_gaps()\n- editorial_subdivide()\nOutput: Finer-grained EGTSegments")
post_subdivide = create_node(POST_X, 620, "3. Subdivide (tasks/scene_detect.py)", "[UNCHANGED]\n- subdivide_by_speech_gaps()\n- editorial_subdivide()\nOutput: Finer-grained EGTSegments")

# Step 4: Analyze
pre_analyze = create_node(PRE_X, 820, "4. Visual Analyze (tasks/analyze.py)", "⚠️ BOTTLENECK: SEQUENTIAL API CALLS\n- Loops 1-by-1 over N segments.\n- Calls Gemini describe_keyframe() N times.\n- Dense sampling for long scenes.\n- synthesize_context() (1 API call)", bg_color="#fee2e2", stroke_color="#ef4444")
post_analyze = create_node(POST_X, 820, "4. Visual Analyze (tasks/analyze.py)", "✅ M4 FIX: PARALLEL BATCHING\n- ThreadPoolExecutor parallel keyframe extraction.\n- Batched Gemini API calls: describe_keyframes_batch().\n- Adaptive dense-sampling floor (20s instead of 10s).\n- synthesize_context()", bg_color="#dcfce7", stroke_color="#22c55e")

# Step 5: Score
pre_score = create_node(PRE_X, 1020, "5. Score (tasks/score.py)", "⚠️ BOTTLENECK: SEQUENTIAL API CALLS\n- Loops 1-by-1 over N segments.\n- Calls classify_egt_segments() N times.\n- Rule-based quality_score & is_bad_take.", bg_color="#fee2e2", stroke_color="#ef4444")
post_score = create_node(POST_X, 1020, "5. Score (tasks/score.py)", "✅ M4 FIX: SLIDING WINDOW BATCHING\n- classify_egt_segments_batch() processes 10-12\n  segments per single Gemini call.\n- Reduces N API calls to N/10 calls.\n- Rule-based quality_score applied after.", bg_color="#dcfce7", stroke_color="#22c55e")

# Step 6: EGT Assembly
pre_egt = create_node(PRE_X, 1220, "6. EGT Assembly (tasks/egt.py)", "Builds & validates EGTDocument.\nStores to jobs_data_db.")
post_egt = create_node(POST_X, 1220, "6. EGT Assembly (tasks/egt.py)", "[UNCHANGED]\nBuilds & validates EGTDocument.\nStores to jobs_data_db.")

# Step 7: EDL Reasoning
pre_edl = create_node(PRE_X, 1420, "7. EDL Reasoning (tasks/edl.py)", "⚠️ BOTTLENECK: CONTEXT OVERFLOW\n- Sends entire 200+ segment EGT to Gemini.\n- generate_edl_llm() single massive API call.\n- Fails on long footage (> 50k tokens).\n- enforce_budget() Tier 3 fallback.", bg_color="#fee2e2", stroke_color="#ef4444", h=180)
post_edl = create_node(POST_X, 1420, "7. EDL Reasoning (tasks/edl.py)", "✅ M4 FIX: MAP-REDUCE REASONING\n- Map: Split EGT into chunks (~35 segs).\n  generate_edl_llm() on each chunk with sub-budget.\n- Reduce: generate_edl_reduce_llm() takes summary\n  of chunks for global sequencing.\n- enforce_budget() Tier 3 fallback.", bg_color="#dcfce7", stroke_color="#22c55e", h=180)

# Step 8: Assemble
pre_assemble = create_node(PRE_X, 1640, "8. Video Assemble (tasks/assemble.py)", "Inputs: EDL, Raw MP4s\n- FFmpeg filtergraph construction\nOutput: Final processed MP4")
post_assemble = create_node(POST_X, 1640, "8. Video Assemble (tasks/assemble.py)", "[UNCHANGED]\nInputs: EDL, Raw MP4s\n- FFmpeg filtergraph construction\nOutput: Final processed MP4")


# Connect nodes
connect(pre_ingest, pre_transcribe)
connect(pre_transcribe, pre_subdivide)
connect(pre_subdivide, pre_analyze)
connect(pre_analyze, pre_score)
connect(pre_score, pre_egt)
connect(pre_egt, pre_edl)
connect(pre_edl, pre_assemble)

connect(post_ingest, post_transcribe)
connect(post_transcribe, post_subdivide)
connect(post_subdivide, post_analyze)
connect(post_analyze, post_score)
connect(post_score, post_egt)
connect(post_egt, post_edl)
connect(post_edl, post_assemble)

# Central line separator
add_element(create_rect(590, 200, 2, 1600, "#94a3b8", "#94a3b8", "dashed", 50))

excalidraw = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {
        "viewBackgroundColor": "#fafafa",
        "gridSize": None
    },
    "files": {}
}

with open("vlogforge-m4-comparison.excalidraw", "w") as f:
    json.dump(excalidraw, f, indent=2)

print("M4 Comparison Excalidraw generated successfully!")
