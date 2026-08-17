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
        "width": len(max(text.split('\n'), key=len)) * (font_size * 0.6),
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

def create_badge(x, y, text, bg_color, text_color):
    group_id = gen_id()
    w = 80
    h = 24
    rect_el = create_rect(x, y, w, h, bg_color, bg_color, roundness=3)
    rect_el["groupIds"] = [group_id]
    
    text_el = create_text(x + 5, y + 4, text, font_size=12, color=text_color, font_family=2, align="center")
    text_el["width"] = w - 10
    text_el["groupIds"] = [group_id]
    
    add_element(rect_el)
    add_element(text_el)

def create_milestone_card(x, y, m_id, title, details, status):
    group_id = gen_id()
    w = 350
    h = 200
    
    # Theme colors based on status
    if status == "DONE":
        bg_color, stroke_color = "#f0fdf4", "#22c55e"
        badge_bg, badge_txt = "#dcfce7", "#064e3b"
    elif status == "NEXT":
        bg_color, stroke_color = "#eff6ff", "#3b82f6"
        badge_bg, badge_txt = "#bfdbfe", "#1e3a8a"
    else:
        bg_color, stroke_color = "#f8fafc", "#94a3b8"
        badge_bg, badge_txt = "#e2e8f0", "#334155"

    rect_el = create_rect(x, y, w, h, bg_color, stroke_color, roundness=3)
    rect_el["groupIds"] = [group_id]
    add_element(rect_el)
    
    # ID / Badge
    create_badge(x + w - 90, y + 10, status, badge_bg, badge_txt)
    
    # Title
    title_el = create_text(x + 15, y + 15, f"{m_id} — {title}", font_size=18, color="#111827", font_family=2, align="left")
    title_el["width"] = w - 110
    title_el["groupIds"] = [group_id]
    add_element(title_el)
    
    # Separator
    sep_el = create_rect(x + 15, y + 50, w - 30, 1, stroke_color, stroke_color, opacity=50)
    sep_el["groupIds"] = [group_id]
    add_element(sep_el)
    
    # Details
    det_el = create_text(x + 15, y + 65, details, font_size=14, color="#4b5563", font_family=3, align="left")
    det_el["width"] = w - 30
    det_el["groupIds"] = [group_id]
    add_element(det_el)
    
    return {"id": rect_el["id"], "x": x, "y": y, "w": w, "h": h}

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

# ------------- LAYOUT -------------

# Header
add_element(create_text(100, 50, "VlogForge Implementation Plan & Milestones", 32, "#0f172a", 2, "left"))
add_element(create_text(100, 100, "Demo-level roadmap of the end-to-end architecture rollout based on MILESTONES.md", 18, "#64748b", 3, "left"))

# Track 1: Completed / Foundation (Y: 200)
add_element(create_rect(50, 200, 1850, 280, "#f8fafc", "#e2e8f0", "dashed"))
add_element(create_text(70, 220, "PHASE 1: FOUNDATION & CORE PIPELINE (DONE)", 20, "#16a34a", 2, "left"))

m0 = create_milestone_card(100, 260, "M0", "Foundation Plumbing", 
    "• Scaffolding, Auth, Project CRUD\n"
    "• Resumable File Upload (tus)\n"
    "• ConnectedAccount concept (YouTube)", "DONE")

m1 = create_milestone_card(550, 260, "M1", "Tier 1 Perception", 
    "• Visual/audio tagging layer\n"
    "• Gemini-powered structured metadata\n"
    "• Activity type, quality signals, notable\n  moments extraction", "DONE")

m2 = create_milestone_card(1000, 260, "M2", "First End-to-End Slice", 
    "• Perception → Reasoning → Assembly\n"
    "• LLM proposes an EDL\n"
    "• Deterministic repair / enforcement\n"
    "• Output final MP4", "DONE")

m3 = create_milestone_card(1450, 260, "M3", "Review UI", 
    "• Human-in-the-loop gate (mandatory)\n"
    "• Creator views/adjusts machine EDL\n"
    "• Final export approval", "DONE")

# Track 2: Upcoming / Scaling (Y: 550)
add_element(create_rect(50, 550, 1850, 280, "#f8fafc", "#e2e8f0", "dashed"))
add_element(create_text(70, 570, "PHASE 2: SCALING & PUBLISHING (NEXT)", 20, "#2563eb", 2, "left"))

m4 = create_milestone_card(100, 610, "M4", "Long-Footage Scaling", 
    "• Chunking/batching perception passes\n"
    "• Scale limits graceful handling\n"
    "• Reasoning tier manages large candidate sets", "NEXT")

m5 = create_milestone_card(550, 610, "M5", "Metadata Generation", 
    "• Auto-gen YouTube titles, desc, tags\n"
    "• Tone & archetype cards\n"
    "• Tappable info icon for AI grounding\n"
    "(Committed stopping point)", "NEXT")

# Track 3: Future / Intelligence (Y: 900)
add_element(create_rect(50, 900, 1850, 280, "#f8fafc", "#e2e8f0", "dashed"))
add_element(create_text(70, 920, "PHASE 3: INTELLIGENCE & PERSONALIZATION (FUTURE)", 20, "#64748b", 2, "left"))

m6 = create_milestone_card(100, 960, "M6", "YT Integration + Vector DB", 
    "• Manual upload of 4-5 published videos\n"
    "• Vector DB stores style exemplars\n"
    "• Learns 'channel brand style' for edits", "FUTURE")

m7 = create_milestone_card(550, 960, "M7", "Vertical-Specific Features", 
    "• Domain logic (lifestyle, gym, travel)\n"
    "• Gym-specific priority handling\n"
    "• Vertical-specific pacing rules", "FUTURE")

m8 = create_milestone_card(1000, 960, "M8", "Creator Profile Learning", 
    "• Quiet, background personalization\n"
    "• Learns catchphrases & structural habits\n"
    "• Invisible profile, improves over time", "FUTURE")


# Connectors (Arrows between phases)
def connect(c1, c2, color="#9ca3af"):
    start_x = c1["x"] + c1["w"]
    start_y = c1["y"] + c1["h"]//2
    end_x = c2["x"]
    end_y = c2["y"] + c2["h"]//2
    create_arrow(start_x, start_y, end_x, end_y, stroke_color=color)

def connect_wrap(c1, c2, color="#9ca3af"):
    # Start from bottom of c1 to top of c2
    start_x = c1["x"] + c1["w"]//2
    start_y = c1["y"] + c1["h"]
    end_x = c2["x"] + c2["w"]//2
    end_y = c2["y"]
    waypoints = [
        (start_x, start_y + 40),
        (50, start_y + 40), # route back to left side
        (50, end_y - 30),
        (end_x, end_y - 30)
    ]
    create_arrow(start_x, start_y, end_x, end_y, waypoints=waypoints, stroke_color=color)

# P1 connections
connect(m0, m1, "#22c55e")
connect(m1, m2, "#22c55e")
connect(m2, m3, "#22c55e")

# P1 -> P2 connection
connect_wrap(m3, m4, "#3b82f6")
connect(m4, m5, "#3b82f6")

# P2 -> P3 connection
connect_wrap(m5, m6, "#94a3b8")
connect(m6, m7, "#94a3b8")
connect(m7, m8, "#94a3b8")


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

with open("vlogforge-implementation-plan.excalidraw", "w") as f:
    json.dump(excalidraw, f, indent=2)

print("Implementation Plan Excalidraw generated successfully!")
