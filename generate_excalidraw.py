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
    return {
        "id": gen_id(),
        "type": "text",
        "x": x,
        "y": y,
        "width": len(max(text.split('\n'), key=len)) * (font_size * 0.6), # Approximation
        "height": len(text.split('\n')) * (font_size * 1.25),
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

def create_group(x, y, w, h, title, subtitle, bg_color, stroke_color, link=None):
    group_id = gen_id()
    
    rect_el = create_rect(x, y, w, h, bg_color, stroke_color, roundness=3)
    rect_el["groupIds"] = [group_id]
    if link:
        rect_el["link"] = link
    
    title_el = create_text(x + 15, y + 15, title, font_size=20, color="#111827", font_family=2, align="left")
    title_el["groupIds"] = [group_id]
    
    sub_el = create_text(x + 15, y + 45, subtitle, font_size=14, color="#4b5563", font_family=3, align="left")
    sub_el["groupIds"] = [group_id]
    
    add_element(rect_el)
    add_element(title_el)
    add_element(sub_el)
    
    return rect_el["id"]

def create_badge(x, y, text, status_color):
    group_id = gen_id()
    w = 90
    h = 28
    rect_el = create_rect(x, y, w, h, status_color, status_color, roundness=3)
    rect_el["groupIds"] = [group_id]
    
    text_el = create_text(x + 5, y + 6, text, font_size=12, color="#064e3b" if status_color == "#dcfce7" else "#713f12", font_family=2, align="center")
    text_el["width"] = w - 10
    text_el["groupIds"] = [group_id]
    
    add_element(rect_el)
    add_element(text_el)

def create_data_object(x, y, title, fields):
    group_id = gen_id()
    w = 220
    h = 40 + len(fields) * 16
    
    rect_el = create_rect(x, y, w, h, "#eff6ff", "#3b82f6", roundness=2)
    rect_el["groupIds"] = [group_id]
    
    title_el = create_text(x + 15, y + 10, title, font_size=16, color="#1e3a8a", font_family=3, align="left")
    title_el["groupIds"] = [group_id]
    
    fields_text = "\n".join(fields)
    fields_el = create_text(x + 15, y + 35, fields_text, font_size=14, color="#3b82f6", font_family=3, align="left")
    fields_el["groupIds"] = [group_id]
    
    add_element(rect_el)
    add_element(title_el)
    add_element(fields_el)
    return rect_el["id"]

def create_arrow(start_x, start_y, end_x, end_y, text="", waypoints=None):
    a_id = gen_id()
    
    # Base points
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
        "strokeColor": "#9ca3af",
        "backgroundColor": "transparent",
        "fillStyle": "hachure",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "roundness": {"type": 2}, # Curvy lines
        "seed": 1,
        "version": 1,
        "versionNonce": 1,
        "isDeleted": False,
        "boundElements": [],
        "points": points,
        "lastCommittedPoint": None,
        "startBinding": None, # Removed binding to prevent Excalidraw from mangling routes
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow"
    }
    
    add_element(arrow_el)
    
    if text:
        # Place text roughly in the middle of the bounding box of the arrow, with a solid white background
        mid_x = start_x + (end_x - start_x)/2
        mid_y = start_y + (end_y - start_y)/2
        if waypoints and len(waypoints) > 0:
            mid_x = waypoints[0][0]
            mid_y = waypoints[0][1]
        
        text_el = create_text(mid_x - 30, mid_y - 15, text, font_size=14, color="#4b5563", font_family=2, align="center", bg_color="#ffffff")
        add_element(text_el)
        
    return a_id


# Colors
C_DONE = "#dcfce7"
C_WARN = "#fef08a"
C_STROKE_DONE = "#22c55e"
C_STROKE_WARN = "#eab308"

# ------------- LAYOUT -------------

# Swimlanes
add_element(create_rect(50, 100, 450, 300, "#f8fafc", "#cbd5e1", "dashed"))
add_element(create_text(70, 120, "1. INGESTION", 20, "#64748b", 2, "left"))

add_element(create_rect(800, 100, 550, 1100, "#f8fafc", "#cbd5e1", "dashed"))
add_element(create_text(820, 120, "2. PERCEPTION (Pass 1)", 20, "#64748b", 2, "left"))

add_element(create_rect(1650, 100, 500, 800, "#f8fafc", "#cbd5e1", "dashed"))
add_element(create_text(1670, 120, "3. REASONING & BUDGET (Pass 2)", 20, "#64748b", 2, "left"))

add_element(create_rect(2450, 100, 450, 300, "#f8fafc", "#cbd5e1", "dashed"))
add_element(create_text(2470, 120, "4. ASSEMBLY (Pass 3)", 20, "#64748b", 2, "left"))

add_element(create_rect(50, 1300, 1300, 400, "#f8fafc", "#cbd5e1", "dashed"))
add_element(create_text(70, 1320, "5. FRONTEND / REVIEW UI", 20, "#64748b", 2, "left"))

add_element(create_rect(1650, 1300, 1250, 400, "#f8fafc", "#cbd5e1", "dashed"))
add_element(create_text(1670, 1320, "6. AUTH & PROJECT MGT", 20, "#64748b", 2, "left"))


# Nodes
create_group(100, 200, 350, 140, "Upload & Extract", "app/tasks/ingest.py\n- Tus Resumable Upload\n- FFmpeg CFR Transcode\n- Audio Extraction", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/backend/app/tasks/ingest.py")
create_badge(340, 215, "DONE", C_DONE)

create_group(850, 200, 450, 120, "Transcription", "app/tasks/transcribe.py\n- Gemini STT / Whisper\n- align_transcript_with_segments()", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/backend/app/tasks/transcribe.py")
create_badge(1185, 215, "DONE", C_DONE)

create_group(850, 450, 450, 140, "Scene Detection", "app/tasks/scene_detect.py\n- subdivide_by_speech_gaps()\n- editorial_subdivide()\n- Dynamic thresholds via target_duration", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/backend/app/tasks/scene_detect.py")
create_badge(1185, 465, "DONE", C_DONE)

create_group(850, 700, 450, 120, "Visual Analysis", "app/tasks/analyze.py\n- analyze_segments()\n- Gemini Vision Keyframe Tags", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/backend/app/tasks/analyze.py")
create_badge(1185, 715, "DONE", C_DONE)

create_group(850, 950, 450, 120, "Quality Scoring & EGT", "app/tasks/score.py | egt.py\n- score_segments() -> is_bad_take\n- build_egt_document()", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/backend/app/tasks/score.py")
create_badge(1185, 965, "DONE", C_DONE)

create_group(1700, 200, 400, 140, "LLM Reasoning (Phase 0)", "app/tasks/edl.py\n- generate_edl_llm()\n- No size-based branching yet\n- _validate_priority_consistency()", C_WARN, C_STROKE_WARN, link="file:///Users/anishshende/vlogforge/backend/app/tasks/edl.py")
create_badge(1995, 215, "PARTIAL", C_WARN)

create_group(1700, 600, 400, 140, "Budget Enforcement", "app/tasks/edl.py\n- _enforce_budget()\n- Tier 3 Graduated Repair\n- Drops LOW/MEDIUM priority", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/backend/app/tasks/edl.py")
create_badge(1995, 615, "DONE", C_DONE)

create_group(2500, 200, 350, 120, "Video Assembly", "app/tasks/assemble.py\n- assemble_vlog()\n- FFmpeg Final Render", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/backend/app/tasks/assemble.py")
create_badge(2745, 215, "DONE", C_DONE)

# Frontend Nodes
create_group(100, 1450, 350, 140, "React SPA Dashboard", "frontend/src/App.jsx\n- UploadPanel.jsx\n- Dashboard.jsx\n- ProcessingMonitor.jsx", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/frontend/src/App.jsx")
create_badge(340, 1465, "DONE", C_DONE)

create_group(850, 1450, 400, 140, "Timeline Editor", "frontend/src/components/VideoPreview.jsx\n- Trigger /api/jobs/{id}/re-reason\n- Trigger /api/jobs/{id}/re-render\n- Timeline mutations", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/frontend/src/components/VideoPreview.jsx")
create_badge(1145, 1465, "DONE", C_DONE)

# Auth Nodes
create_group(1700, 1450, 350, 140, "JWT Auth & API", "app/routers/auth.py\napp/main.py\n- User Registration & Login\n- WebSocket /ws/{job_id}", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/backend/app/routers/auth.py")
create_badge(1945, 1465, "DONE", C_DONE)

create_group(2400, 1450, 350, 140, "PostgreSQL Store", "app/db_models.py\n- User Model\n- Project Model\n- VideoFile Model", C_DONE, C_STROKE_DONE, link="file:///Users/anishshende/vlogforge/backend/app/db_models.py")
create_badge(2645, 1465, "DONE", C_DONE)


# Data Objects (Floating on arrows)
create_data_object(550, 215, "Raw EGTSegments", ["+ clip_id", "+ start_sec", "+ end_sec", "+ source_file"])
create_data_object(1400, 500, "EGTDocument", ["+ segments[]", "+ is_bad_take", "+ quality_score", "+ tags / visual_desc"])
create_data_object(1750, 415, "EDLEntry[]", ["+ clip_id", "+ start/end_sec", "+ narrative_priority", "+ sequence_index"])
create_data_object(2200, 215, "Trimmed EDLEntry[]", ["+ Fits target_duration", "+ Padded/Trimmed"])


# Draw Arrows (Manually routed to prevent overlaps)

# Ingest -> Transcribe (via Data Object)
create_arrow(450, 260, 550, 260)
create_arrow(770, 260, 850, 260)

# Perception Downward Flow
create_arrow(1075, 320, 1075, 450, "Aligned Segments")
create_arrow(1075, 590, 1075, 700, "Refined Segments")
create_arrow(1075, 820, 1075, 950, "Described Segments")

# Perception -> Reasoning (via EGT Data Object)
create_arrow(1300, 1010, 1400, 550, "", waypoints=[(1350, 1010), (1350, 550)])
create_arrow(1620, 550, 1700, 260, "", waypoints=[(1660, 550), (1660, 260)])

# Reasoning -> Budget (via EDL Data Object)
create_arrow(1900, 340, 1900, 415)
create_arrow(1900, 515, 1900, 600)

# Budget -> Assembly (via Final EDL Data Object)
create_arrow(2100, 660, 2200, 260, "", waypoints=[(2150, 660), (2150, 260)])
create_arrow(2420, 260, 2500, 260)

# Assembly -> Frontend (Timeline)
create_arrow(2675, 320, 1050, 1450, "Final MP4 Render", waypoints=[(2675, 1250), (1050, 1250)])

# Frontend internal
create_arrow(450, 1510, 850, 1510, "Job Status")

# Frontend -> Reasoning (Re-reason)
create_arrow(1000, 1450, 1750, 340, "Trigger /api/.../re-reason", waypoints=[(1000, 1375), (1500, 1375), (1500, 380), (1750, 380)])

# Frontend -> Assembly (Re-render)
create_arrow(1150, 1450, 2600, 320, "Trigger /api/.../re-render", waypoints=[(1150, 1350), (2300, 1350), (2300, 350), (2600, 350)])

# Auth -> DB
create_arrow(2050, 1510, 2400, 1510, "CRUD Projects")


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

with open("vlogforge-architecture.excalidraw", "w") as f:
    json.dump(excalidraw, f, indent=2)

print("Advanced Excalidraw generated successfully!")
