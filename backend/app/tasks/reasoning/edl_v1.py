import logging
from typing import List, Dict, Optional
from app.models import EDLItem

logger = logging.getLogger("VlogForge.EDL")

def is_background_noise(text: str, visual: str, user_prompt: str) -> bool:
    """Detects if a segment's audio/transcript represents unwanted background noise, static, or announcements."""
    import re
    text_lower = text.lower().strip()
    visual_lower = visual.lower().strip()
    user_prompt_lower = user_prompt.lower()
    
    # 1. If user explicitly requests this context (e.g. plane, flight, train, bells), do not classify as noise
    contributing_keywords = ["plane", "flight", "announcement", "train", "bell", "chime", "siren", "noise", "static", "alarm", "music", "laughter", "airport", "station", "warning"]
    if any(ck in user_prompt_lower for ck in contributing_keywords):
        return False
        
    # 2. General background noise and non-speech indicator patterns (e.g., [music], (chime))
    bracket_noise = re.search(r'\[(music|laughter|applause|chime|sigh|cough|throat|static|hum|buzz|beep|siren|alarm|noise|whisper|murmur|chattering|screech)\]', text_lower)
    if bracket_noise:
        return True
        
    # 3. Comprehensive noise, warning, and PA announcement keywords across industries
    noise_keywords = [
        # Transit/Public PA announcements
        "announcement", "cabin crew", "safety card", "seat belt", "seatbelt", "tali keledar", "tempat duduk", "fasten", "attention passengers", "now boarding", "final call", "flight departure", "train station", "subway chime",
        # Ambient/Acoustic noise
        "wind noise", "background noise", "ambient chatter", "loud static", "hissing", "buzzing", "muffled audio", "unintelligible", "microphone rustle", "distortion", "feedback loop",
        # Generic alerts/signals
        "alarm clock", "siren wailing", "smoke detector", "beeping sound", "ringtone", "bell chime", "buzzer sound", "car honking", "traffic noise", "construction noise", "drilling sound", "dog barking"
    ]
    
    if any(nk in text_lower for nk in noise_keywords):
        return True
        
    if any(nk in visual_lower for nk in ["noisy background", "loud hum", "muffled sound", "announcement playing"]):
        return True
        
    return False

def apply_genre_scoring_boost(scenes: List[Dict], genre: str, user_prompt: str = "") -> List[Dict]:
    """Applies a genre-based score boost to segments based on visual and transcript keyword matches,
    and penalizes background noise unless requested by the user.
    """
    boosted_scenes = []
    
    gym_keywords = ["barbell", "weight", "lifting", "dumbbell", "bench", "deadlift", "squat", "pullup", "gym", "workout", "set", "rep", "exercise", "training", "fitness"]
    travel_keywords = ["landscape", "nature", "view", "market", "street", "travel", "explore", "trip", "landmark", "sunset", "beach", "mountain", "outdoor", "beautiful", "scenery", "hotel", "city"]
    makeup_keywords = ["face", "eye", "lips", "cosmetics", "palette", "brush", "makeup", "apply", "mirror", "tutorial", "lipstick", "skin", "foundation", "shade", "product"]
    daily_keywords = ["routine", "lifestyle", "vlog", "home", "room", "apartment", "living", "day", "talk", "chat", "vlogging", "personal", "morning", "night", "kitchen", "cooking", "eating", "food"]

    for scene in scenes:
        s_copy = scene.copy()
        text = s_copy.get("text", "").lower()
        visual = s_copy.get("visual_description", "").lower()
        
        boost = 0.0
        if genre == "gym":
            has_keyword = any(kw in text or kw in visual for kw in gym_keywords)
            if has_keyword:
                boost = 0.35
            if s_copy.get("label") == "B_ROLL":
                boost += 0.15
        elif genre == "travel":
            has_keyword = any(kw in text or kw in visual for kw in travel_keywords)
            if has_keyword:
                boost = 0.35
            if s_copy.get("label") == "B_ROLL":
                boost += 0.20  # Boost travel B-rolls more
        elif genre == "makeup":
            has_keyword = any(kw in text or kw in visual for kw in makeup_keywords)
            if has_keyword:
                boost = 0.35
            if s_copy.get("label") == "HIGHLIGHT":
                boost += 0.10
        elif genre == "daily":
            has_keyword = any(kw in text or kw in visual for kw in daily_keywords)
            if has_keyword:
                boost = 0.35
            if s_copy.get("label") == "HIGHLIGHT":
                boost += 0.15  # Boost daily lifestyle highlights/talking
                
        # Calculate final score with boost.
        # float() guard: Gemini occasionally returns score as a JSON string ("0.9" vs 0.9).
        # Without this cast, str + float raises TypeError when mock or LLM returns a string score.
        new_score = min(1.0, float(s_copy.get("score", 1.0)) + boost)
        
        # Avoid background noise/announcements unless contributing
        if is_background_noise(text, visual, user_prompt):
            new_score = max(0.0, new_score - 0.40)
            
        s_copy["score"] = new_score
        boosted_scenes.append(s_copy)
        
    return boosted_scenes

def snap_boundary_to_speech(start: float, end: float, video_file: str, transcript_segments: List[Dict]) -> tuple:
    """Snaps the start and end of a clip to aligned speech segment boundaries if they are close,
    preventing clipping mid-sentence or mid-word.
    """
    if not transcript_segments:
        return start, end
        
    snapped_start = start
    snapped_end = end
    
    file_segs = [t for t in transcript_segments if t.get("video_file") == video_file]
    if not file_segs:
        return start, end
        
    # Snap start: find the speech segment that starts close to the scene start [start - 1.0, start + 1.5]
    best_start_diff = 1.5
    for seg in file_segs:
        diff = abs(seg["start"] - start)
        if diff < best_start_diff:
            if seg["start"] < end - 0.5:
                snapped_start = seg["start"]
                best_start_diff = diff
                
    # Snap end: find the speech segment that ends close to the scene end [end - 1.5, end + 1.0]
    best_end_diff = 1.5
    for seg in file_segs:
        diff = abs(seg["end"] - end)
        if diff < best_end_diff:
            if seg["end"] > snapped_start + 0.5:
                snapped_end = seg["end"]
                best_end_diff = diff
                
    return snapped_start, snapped_end

def generate_edl(
    classified_scenes: List[Dict],
    total_raw_duration: float,
    target_duration_sec: Optional[float] = None,
    context_doc: str = "",
    user_prompt: str = "",
    vlog_genre: str = "default",
    transcript_segments: Optional[List[Dict]] = None
) -> List[Dict]:
    """Assemble the Edit Decision List (EDL) based on classification, genre weights, and pacing rules."""
    logger.info(f"Generating Edit Decision List (EDL) for genre: {vlog_genre}...")
    
    # Apply genre-specific score boost and noise suppression
    classified_scenes = apply_genre_scoring_boost(classified_scenes, vlog_genre, user_prompt)
    
    # 1. Determine Target Duration
    if target_duration_sec and target_duration_sec > 0:
        target_duration = min(target_duration_sec, total_raw_duration)
    else:
        if total_raw_duration >= 600.0:  # 10 minutes
            target_duration = 600.0
        else:
            target_duration = max(30.0, total_raw_duration * 0.30) # min 30s or 30%
        
    logger.info(f"Total raw duration: {total_raw_duration:.2f}s. Target duration: {target_duration:.2f}s.")
    
    # Tag copies to track chronological index
    scenes_with_indices = []
    for idx, s in enumerate(classified_scenes):
        s_copy = s.copy()
        s_copy["_orig_idx"] = idx
        scenes_with_indices.append(s_copy)

    is_reel = target_duration_sec is not None and target_duration_sec <= 60.0
    
    # Dynamic pacing parameters based on genre
    if vlog_genre == "gym":
        max_talking_limit = 25.0
    elif vlog_genre == "travel":
        max_talking_limit = 60.0
    elif vlog_genre == "makeup":
        max_talking_limit = 120.0
    elif vlog_genre == "daily":
        max_talking_limit = 80.0
    else:
        max_talking_limit = 90.0
        
    if is_reel:
        logger.info("Reel Option detected. Running reel selection strategy.")
        from app.utils.llm import select_reel_segments_llm
        llm_selected = select_reel_segments_llm(scenes_with_indices, target_duration, context_doc, user_prompt)
        
        if llm_selected is not None:
            # Enforce that INTRO segments are always placed at the front of the EDL, and OUTROs at the end
            intros = [s for s in llm_selected if s.get("label") == "INTRO"]
            outros = [s for s in llm_selected if s.get("label") == "OUTRO"]
            middle = [s for s in llm_selected if s.get("label") not in ["INTRO", "OUTRO"]]
            sequenced_edl = intros + middle + outros
            current_duration = sum(s["end"] - s["start"] for s in sequenced_edl)
        else:
            # Fallback Python heuristic reel selection
            logger.info("Running Python fallback reel selection...")
            candidates = [s for s in scenes_with_indices if s["label"] != "FILLER"]
            
            candidates_by_file = {}
            for c in candidates:
                file_name = c["video_file"]
                if file_name not in candidates_by_file:
                    candidates_by_file[file_name] = []
                candidates_by_file[file_name].append(c)
                
            for file_name in candidates_by_file:
                candidates_by_file[file_name].sort(key=lambda x: x.get("score", 1.0), reverse=True)
                
            selected_scenes = []
            current_duration = 0.0
            
            # Round 1: Best snippet from every video
            file_order = sorted(candidates_by_file.keys())
            for file_name in file_order:
                if candidates_by_file[file_name]:
                    best_scene = candidates_by_file[file_name][0]
                    best_scene_dur = best_scene["end"] - best_scene["start"]
                    selected_scenes.append(best_scene)
                    current_duration += best_scene_dur
                    
            # Round 2: Fill remaining target duration budget
            remaining_candidates = []
            used_keys = set((s["video_file"], s["start"], s["end"]) for s in selected_scenes)
            for file_name in candidates_by_file:
                for c in candidates_by_file[file_name]:
                    key = (c["video_file"], c["start"], c["end"])
                    if key not in used_keys:
                        remaining_candidates.append(c)
                        
            remaining_candidates.sort(key=lambda x: x.get("score", 1.0), reverse=True)
            
            for c in remaining_candidates:
                if current_duration >= target_duration:
                    break
                dur = c["end"] - c["start"]
                # Enforce +/- 10s per minute limit (maximum threshold budget window)
                if current_duration + dur > target_duration + (10.0 * (target_duration / 60.0)):
                    continue
                selected_scenes.append(c)
                current_duration += dur
                
            # Chronological fallback sort
            selected_scenes.sort(key=lambda x: x["_orig_idx"])
            
            # Sequence fallback selection using LLM
            from app.utils.llm import sequence_edl_segments
            sequenced_edl = sequence_edl_segments(selected_scenes, context_doc)
            
            # Enforce that INTRO segments are always placed at the front of the EDL, and OUTROs at the end
            intros = [s for s in sequenced_edl if s.get("label") == "INTRO"]
            outros = [s for s in sequenced_edl if s.get("label") == "OUTRO"]
            middle = [s for s in sequenced_edl if s.get("label") not in ["INTRO", "OUTRO"]]
            sequenced_edl = intros + middle + outros
            
            current_duration = sum(s["end"] - s["start"] for s in sequenced_edl)
    else:
        # Group segments by label
        intros = [s for s in scenes_with_indices if s["label"] == "INTRO"]
        outros = [s for s in scenes_with_indices if s["label"] == "OUTRO"]
        highlights = [s for s in scenes_with_indices if s["label"] == "HIGHLIGHT"]
        b_rolls = [s for s in scenes_with_indices if s["label"] == "B_ROLL"]
        
        # Sort candidates by score descending
        intros.sort(key=lambda x: x.get("score", 1.0), reverse=True)
        outros.sort(key=lambda x: x.get("score", 1.0), reverse=True)
        highlights.sort(key=lambda x: x.get("score", 1.0), reverse=True)
        b_rolls.sort(key=lambda x: x.get("score", 1.0), reverse=True)
        
        edl_list = []
        current_duration = 0.0
        
        # 2. Add Intro first
        selected_intro = None
        if intros:
            selected_intro = intros[0]
            intro_dur = selected_intro["end"] - selected_intro["start"]
            edl_list.append(selected_intro)
            current_duration += intro_dur
            logger.info(f"Selected INTRO: {selected_intro['video_file']} ({selected_intro['start']:.2f}s - {selected_intro['end']:.2f}s)")
            
        # 3. Reserve Outro duration
        selected_outro = None
        reserved_outro_dur = 0.0
        if outros:
            selected_outro = outros[0]
            reserved_outro_dur = selected_outro["end"] - selected_outro["start"]
            logger.info(f"Reserved OUTRO: {selected_outro['video_file']} ({selected_outro['start']:.2f}s - {selected_outro['end']:.2f}s)")
            
        # 4. Fill the middle greedily
        unused_intros = [i for i in intros if i != selected_intro]
        unused_outros = [o for o in outros if o != selected_outro]
        
        middle_candidates = highlights + b_rolls
        for item in unused_intros + unused_outros:
            item_copy = item.copy()
            item_copy["label"] = "HIGHLIGHT"
            middle_candidates.append(item_copy)
            
        # Sort overall middle candidates by score
        middle_candidates.sort(key=lambda x: x.get("score", 1.0), reverse=True)
        
        selected_middle = []
        consecutive_talking_dur = 0.0
        
        # We loop until we reach target_duration minus reserved_outro_dur
        middle_target = target_duration - current_duration - reserved_outro_dur
        logger.info(f"Target duration for middle segments: {middle_target:.2f}s")
        
        used_keys = set()
        if selected_intro:
            used_keys.add((selected_intro["video_file"], selected_intro["start"], selected_intro["end"]))
        if selected_outro:
            used_keys.add((selected_outro["video_file"], selected_outro["start"], selected_outro["end"]))
            
        while current_duration < (target_duration - reserved_outro_dur) and middle_candidates:
            # Find next best segment respecting pacing limit
            best_candidate_idx = -1
            
            for idx, candidate in enumerate(middle_candidates):
                key = (candidate["video_file"], candidate["start"], candidate["end"])
                if key in used_keys:
                    continue
                    
                is_talking = candidate["label"] == "HIGHLIGHT"
                
                # If we've been talking for more than max_talking_limit, prefer a B-roll
                if consecutive_talking_dur >= max_talking_limit and is_talking:
                    # Find a B-roll if one exists
                    has_broll = any(c["label"] == "B_ROLL" and (c["video_file"], c["start"], c["end"]) not in used_keys for c in middle_candidates)
                    if has_broll:
                        continue
                
                best_candidate_idx = idx
                break
                
            if best_candidate_idx == -1:
                break
                
            candidate = middle_candidates.pop(best_candidate_idx)
            key = (candidate["video_file"], candidate["start"], candidate["end"])
            
            seg_dur = candidate["end"] - candidate["start"]
            
            # Enforce +/- 10s per minute limit (maximum budget window check)
            max_allowed = target_duration + (10.0 * (target_duration / 60.0))
            if current_duration + seg_dur + reserved_outro_dur > max_allowed:
                continue
                
            used_keys.add(key)
            selected_middle.append(candidate)
            current_duration += seg_dur
            
            # Track pacing
            if candidate["label"] == "HIGHLIGHT":
                consecutive_talking_dur += seg_dur
            else:
                consecutive_talking_dur = 0.0  # reset talking on B-roll
                
            logger.info(f"Selected MID: {candidate['video_file']} [{candidate['label']}] ({candidate['start']:.2f}s - {candidate['end']:.2f}s), current total: {current_duration:.2f}s")
            
        # Sort selected middle segments in chronological fallback order
        selected_middle.sort(key=lambda x: x["_orig_idx"])
        logger.info("Sorted selected middle segments in chronological fallback order.")
        
        # Assemble raw default edl list
        raw_edl = []
        if selected_intro:
            raw_edl.append(selected_intro)
        raw_edl.extend(selected_middle)
        if selected_outro:
            raw_edl.append(selected_outro)
            
        # Attempt LLM sequencing based on narrative cues and transcripts
        from app.utils.llm import sequence_edl_segments
        sequenced_edl = sequence_edl_segments(raw_edl, context_doc)
        
    # Format to required output EDL list with boundary cut snapping
    formatted_edl = []
    for s in sequenced_edl:
        start_sec = s["start"]
        end_sec = s["end"]
        
        # Snap start/end to nearest speech segments if transcript_segments is provided
        if transcript_segments:
            start_sec, end_sec = snap_boundary_to_speech(start_sec, end_sec, s["video_file"], transcript_segments)
            
        formatted_edl.append({
            "video_file": s["video_file"],
            "start_sec": start_sec,
            "end_sec": end_sec,
            "type": s["label"]
        })
        
    logger.info(f"EDL generation complete. Created {len(formatted_edl)} cuts. Total duration: {current_duration:.2f}s")
    return formatted_edl
