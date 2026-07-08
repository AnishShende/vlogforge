import os
import json
import logging
import time
import base64
import re
import pydantic
from functools import wraps
from typing import List, Dict, Optional
from PIL import Image
from app.config import settings

logger = logging.getLogger("VlogForge.LLM")

_gemini_client = None

def init_gemini() -> bool:
    global _gemini_client
    if _gemini_client is not None:
        return True
    
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set. LLM features will run in Mock Mode.")
        return False
        
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini API Client successfully initialized.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API Client: {e}. Falling back to Mock Mode.")
        return False

def with_gemini_retry(max_retries=5, base_delay=5.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                    is_daily_quota = "quota" in err_str.lower() and ("perday" in err_str.lower() or "free_tier_requests" in err_str.lower())
                    
                    if is_daily_quota:
                        logger.error("Gemini API daily quota limit exceeded. Failing immediately to trigger model fallback.")
                        raise e
                        
                    if is_rate_limit and attempt < max_retries:
                        delay_match = re.search(r'retryDelay.*?(\d+)', err_str)
                        retry_delay = float(delay_match.group(1)) if delay_match else (base_delay * (2 ** attempt))
                        logger.warning(
                            f"Gemini API rate limit hit in {func.__name__} (attempt {attempt+1}/{max_retries}). "
                            f"Waiting {retry_delay:.1f}s before retry..."
                        )
                        time.sleep(retry_delay)
                        attempt += 1
                    else:
                        raise e
            return func(*args, **kwargs)
        return wrapper
    return decorator

@with_gemini_retry(max_retries=5)
def safe_generate_content(*args, **kwargs):
    if not init_gemini():
        raise RuntimeError("Gemini not initialized")
    
    # Extract model name
    model = kwargs.get("model")
    if not model and len(args) > 0:
        model = args[0]
        
    # Priority list of models to fall back to if daily quota is exceeded or model is missing
    fallbacks = [
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest"
    ]
    
    if model not in fallbacks:
        return _gemini_client.models.generate_content(*args, **kwargs)
        
    start_idx = fallbacks.index(model)
    
    for idx in range(start_idx, len(fallbacks)):
        current_model = fallbacks[idx]
        
        # Modify the arguments to use the current fallback model
        current_kwargs = kwargs.copy()
        current_args = list(args)
        if "model" in current_kwargs:
            current_kwargs["model"] = current_model
        elif len(current_args) > 0:
            current_args[0] = current_model
            
        try:
            return _gemini_client.models.generate_content(*current_args, **current_kwargs)
        except Exception as e:
            err_str = str(e)
            is_daily_quota = "quota" in err_str.lower() and ("perday" in err_str.lower() or "free_tier_requests" in err_str.lower())
            is_not_found = "404" in err_str or "not_found" in err_str.lower()
            if (is_daily_quota or is_not_found) and idx < len(fallbacks) - 1:
                logger.warning(
                    f"Model {current_model} failed ({'daily quota exceeded' if is_daily_quota else 'not found'}). "
                    f"Automatically falling back to {fallbacks[idx+1]}..."
                )
                continue
            raise e
            
    # Fallback to the original model call if somehow loop terminates without raising (failsafe)
    return _gemini_client.models.generate_content(*args, **kwargs)

def describe_keyframe(image_path: str, context_notes: str = "") -> str:
    """Describe a keyframe image using Gemini Multimodal, or return a mock description."""
    if not init_gemini():
        # Mock description based on file location
        filename = os.path.basename(image_path)
        return f"Mock visual: Visual scene from keyframe {filename}. Shows a vlogger setting up their shot with active movement."

    try:
        # Load image using PIL
        img = Image.open(image_path)
        
        prompt = (
            "Describe what is happening in this keyframe image from a video vlog. "
            "Keep the description concise (1-2 sentences), focusing on subjects, lighting, "
            "actions, and visual interest. "
        )
        if context_notes:
            prompt += f"Context notes for the vlog: {context_notes}"
            
        response = safe_generate_content(
            model="gemini-flash-lite-latest",
            contents=[prompt, img]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini keyframe description failed: {e}. Falling back to mock.")
        return "Visual: Vlogger in frame, speaking directly to the camera with soft indoor lighting."

def synthesize_context(transcripts: List[Dict], visual_descriptions: List[str], user_context: str = "") -> str:
    """Summarize the transcripts, keyframes, and user context into a coherent topic/mood document."""
    if not init_gemini():
        return (
            f"Vlog Theme: General Vlog\n"
            f"Mood: Energetic & Engaging\n"
            f"Key Subject: Vlogger sharing experiences.\n"
            f"User Context: {user_context or 'None provided'}"
        )

    try:
        prompt = (
            "You are an expert video producer. Synthesize a 'Context Document' that summarizes "
            "the topic, overall mood, key subjects, and pacing of the vlog based on the following inputs:\n\n"
            f"User context notes: {user_context}\n\n"
            f"Sample transcript snippets: {json.dumps(transcripts[:10])}\n\n"
            f"Visual descriptions: {json.dumps(visual_descriptions[:10])}\n\n"
            "Provide a short, structured summary (Topic, Mood, Notable moments, Main subjects)."
        )
        
        response = safe_generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Context synthesis failed: {e}. Returning fallback.")
        return f"Vlog Context Summary. User Context: {user_context}. Pacing: Standard vlog. Focus: Dynamic scenes."

def build_rolling_window_summary(scenes: List[Dict], current_index: int, window_sec: float = 180.0) -> str:
    """Build a lightweight text summary of the preceding `window_sec` seconds of transcript content.
    
    Used by classify_segments to provide a rolling local context to the LLM, preventing context
    drift when a vlog shifts topics mid-video. Only text content is included to keep the summary
    compact and within token budgets.
    
    Args:
        scenes: Full ordered list of all scene segments (must be in chronological order).
        current_index: Index of the segment currently being classified.
        window_sec: Rolling window size in seconds (default: 180s = 3 minutes).
    
    Returns:
        A concatenated string of transcript text from the preceding window, or an empty string.
    """
    if current_index <= 0 or not scenes:
        return ""
    
    current_start = scenes[current_index].get("start", scenes[current_index].get("start_sec", 0.0))
    window_start = current_start - window_sec
    
    window_texts = []
    for i in range(current_index):
        seg = scenes[i]
        seg_end = seg.get("end", seg.get("end_sec", 0.0))
        seg_text = seg.get("text", seg.get("transcript", "")).strip()
        # Include segment if it falls within the rolling window and has speech content
        if seg_end >= window_start and seg_text:
            window_texts.append(seg_text)
    
    return " ".join(window_texts) if window_texts else ""


def classify_segments(context_doc: str, scenes: List[Dict], full_scenes: Optional[List[Dict]] = None) -> List[Dict]:
    """Classify each scene segment into INTRO, OUTRO, HIGHLIGHT, FILLER, or B_ROLL.
    
    When full_scenes is provided, each segment is classified using a Dynamic Context Tuple:
      - Global Ruleset: The user prompt and global context document.
      - Rolling Window: Transcript text from the preceding 180 seconds.
      - Target Segment: The single segment being evaluated.
    This prevents context drift in long unscripted vlogs that shift topics mid-video.
    """
    if not init_gemini():
        return _mock_classify_segments(scenes)

    # The timeline for rolling window lookups — prefer full_scenes if supplied
    timeline = full_scenes if full_scenes else scenes

    try:
        from google.genai import types

        classified_results = []

        for seg_idx, scene in enumerate(scenes):
            # Build rolling window context for this segment
            # Find the index of this scene in the timeline by matching start/end timestamps
            timeline_index = seg_idx
            if full_scenes:
                for t_idx, t_scene in enumerate(timeline):
                    if (
                        abs(t_scene.get("start", -1) - scene.get("start", -2)) < 0.01
                        and t_scene.get("video_file") == scene.get("video_file")
                    ):
                        timeline_index = t_idx
                        break

            rolling_window = build_rolling_window_summary(timeline, timeline_index, window_sec=180.0)

            rolling_context_block = (
                f"Rolling Context (preceding 3 minutes of transcript):\n"
                f"{rolling_window if rolling_window else '[No preceding content — this is early in the vlog.]'}\n\n"
            )

            prompt = (
                "You are a professional YouTube video editor. Classify the following single video scene segment "
                "into one of these categories:\n"
                "1. INTRO: Opening greeting, introduction, welcoming the viewers.\n"
                "2. OUTRO: Closing thoughts, subscription call-to-action, goodbyes.\n"
                "3. HIGHLIGHT: High-energy, visually rich, topic-essential, OR narrative/timeline transition setup "
                "statements (where the speaker explains what they will show next) that should be in the final video.\n"
                "4. FILLER: Dead air, pauses, mistakes, setup time, repetitive speech, boring sections, "
                "non-speech sounds (e.g. coughing, static, hums), or segments the user context explicitly asks to cut.\n"
                "5. B_ROLL: Scenic, contextual, or transitioning footage with minimal speech.\n\n"
                "CLASSIFICATION RULES:\n"
                "- Use the GLOBAL RULESET to apply the user's overarching goals and any explicit include/exclude instructions.\n"
                "- Use the ROLLING CONTEXT to determine whether the segment logically continues or naturally flows "
                "from recent prior content. A topic that was not set up in the global context but IS present in the "
                "rolling context should NOT be classified as FILLER — it is a valid narrative continuation.\n"
                "- Do NOT classify narrative transition cues (e.g. 'first let me show you X, then we will see Y') as FILLER.\n"
                "- Do NOT classify ambient background noise segments as FILLER; use B_ROLL or HIGHLIGHT with lower score (0.5–0.6).\n"
                "- If the Context Document asks to cut or omit certain speakers/topics, label them FILLER (score 1.0).\n\n"
                f"=== GLOBAL RULESET ===\n{context_doc}\n\n"
                f"=== {rolling_context_block}"
                f"=== SEGMENT TO CLASSIFY ===\n{json.dumps(scene)}\n\n"
                "Output a single JSON object with exactly these keys: "
                "'video_file', 'start', 'end', 'label', 'score', 'text', 'visual_description'. "
                "Return ONLY the JSON object."
            )

            try:
                response = safe_generate_content(
                    model="gemini-flash-lite-latest",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                result_text = response.text.strip()
                seg_result = json.loads(result_text)
                # Handle if model returns a list instead of a single object
                if isinstance(seg_result, list) and seg_result:
                    seg_result = seg_result[0]
                # Coerce score to float defensively (Gemini can return "0.9" as string)
                if "score" in seg_result:
                    try:
                        seg_result["score"] = float(seg_result["score"])
                    except (ValueError, TypeError):
                        seg_result["score"] = 0.75
                classified_results.append(seg_result)
            except Exception as seg_err:
                logger.warning(
                    f"Rolling classification failed for segment {seg_idx} "
                    f"({scene.get('video_file')} @ {scene.get('start')}s): {seg_err}. "
                    "Using rule-based fallback for this segment."
                )
                classified_results.extend(_mock_classify_segments([scene]))

        return classified_results

    except Exception as e:
        logger.error(f"LLM segment classification failed: {e}. Falling back to rule-based mock classification.")
        return _mock_classify_segments(scenes)

def _mock_classify_segments(scenes: List[Dict]) -> List[Dict]:
    """Fallback rule-based segment classification for offline or failed LLM runs."""
    classified = []
    
    # Heuristics keywords
    intro_keywords = ["hi", "hello", "welcome", "hey guys", "what's up", "good morning", "starting", "today we"]
    outro_keywords = ["bye", "see you", "subscribe", "thanks for watching", "outro", "peace out", "next time", "thats it"]
    filler_keywords = ["wait a minute", "hang on", "pause", "re-do", "mistake"]
    
    def contains_keyword(text_val: str, keywords: list) -> bool:
        text_lower = text_val.lower()
        # Find all words
        words = set(re.findall(r'\b\w+\b', text_lower))
        for kw in keywords:
            kw_lower = kw.lower()
            if " " in kw_lower:
                if kw_lower in text_lower:
                    return True
            else:
                if kw_lower in words:
                    return True
        return False

    for scene in scenes:
        text = scene.get("text", "")
        video_file = scene.get("video_file", "")
        start = scene.get("start", 0.0)
        end = scene.get("end", 0.0)
        v_desc = scene.get("visual_description", "")
        
        # Content checks using robust word-boundary matching
        is_intro = contains_keyword(text, intro_keywords)
        is_outro = contains_keyword(text, outro_keywords)
        
        # Check if segment consists entirely of disfluencies/filler words or has actual mistakes
        words = set(re.findall(r'\b\w+\b', text.lower()))
        is_only_disfluencies = len(words) > 0 and words.issubset({"uh", "um"})
        is_filler = is_only_disfluencies or contains_keyword(text, filler_keywords) or (end - start < 1.5 and len(text) == 0)
        
        if is_intro:
            label = "INTRO"
            score = 0.9
        elif is_outro:
            label = "OUTRO"
            score = 0.9
        elif is_filler:
            label = "FILLER"
            score = 0.8
        elif len(text) < 5 and len(v_desc) > 0 and ("landscape" in v_desc.lower() or "b-roll" in v_desc.lower() or "view" in v_desc.lower() or "scenic" in v_desc.lower()):
            label = "B_ROLL"
            score = 0.85
        else:
            label = "HIGHLIGHT"
            score = 0.75
            
        classified.append({
            "video_file": video_file,
            "start": start,
            "end": end,
            "label": label,
            "score": score,
            "text": text,
            "visual_description": v_desc
        })
        
    return classified

def transcribe_audio_gemini(audio_path: str) -> List[Dict]:
    """Transcribe an audio file using Gemini 2.5 Flash STT."""
    if not init_gemini():
        logger.warning("Gemini not initialized. Skipping Gemini STT.")
        return []

    uploaded_file = None
    try:
        from google.genai import types
        logger.info(f"Uploading audio file to Gemini API: {audio_path}...")
        
        # Upload the audio file to Gemini File API
        uploaded_file = _gemini_client.files.upload(file=audio_path)
        logger.info(f"Audio file uploaded successfully. File URI: {uploaded_file.uri}")
        
        prompt = (
            "Transcribe the audio file. Return a JSON list of segments. "
            "Each segment in the list MUST be a dictionary containing exactly these keys:\n"
            "- 'start': start time of speech in seconds (as a float, e.g. 1.25)\n"
            "- 'end': end time of speech in seconds (as a float, e.g. 4.80)\n"
            "- 'text': the transcribed speech string (concise and cleaned)\n\n"
            "CRITICAL:\n"
            "- Cover the entire duration of the audio.\n"
            "- Ensure the timestamps are strictly chronological.\n"
            "- Output only the JSON list of objects."
        )
        
        logger.info("Requesting transcription from Gemini 1.5 Flash Lite...")
        response = safe_generate_content(
            model="gemini-flash-lite-latest",
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        result_text = response.text.strip()
        data = json.loads(result_text)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "segments" in data:
            return data["segments"]
        else:
            raise ValueError("Unexpected JSON format from Gemini STT")
            
    except Exception as e:
        logger.error(f"Gemini 1.5 Flash Lite STT transcription failed: {e}")
        return []
    finally:
        if uploaded_file is not None:
            try:
                logger.info(f"Cleaning up uploaded file from Gemini File API: {uploaded_file.name}...")
                _gemini_client.files.delete(name=uploaded_file.name)
            except Exception as delete_err:
                logger.warning(f"Failed to delete uploaded file from Gemini File API: {delete_err}")


def sequence_edl_segments(scenes: List[Dict], context_doc: str) -> List[Dict]:
    """Use Gemini to sequence the selected EDL segments into a logically coherent narrative flow."""
    if not init_gemini():
        logger.warning("Gemini not initialized. Skipping LLM sequencing, keeping chronological fallback.")
        intros = [s for s in scenes if s.get("label") == "INTRO"]
        outros = [s for s in scenes if s.get("label") == "OUTRO"]
        middle = [s for s in scenes if s.get("label") not in ["INTRO", "OUTRO"]]
        return intros + middle + outros

    try:
        from google.genai import types
        
        # We separate INTRO, OUTRO, and middle segments so the LLM only sequences the middle ones
        intros = [s for s in scenes if s.get("label") == "INTRO"]
        outros = [s for s in scenes if s.get("label") == "OUTRO"]
        middle = [s for s in scenes if s.get("label") not in ["INTRO", "OUTRO"]]
        
        if not middle:
            return scenes
            
        prompt = (
            "You are an expert video editor. You are given a list of video clips (middle segments) "
            "that have been selected for a final vlog. Your task is to rearrange the order of these clips "
            "so that they form a logically coherent, chronological, and narrative story.\n\n"
            "Instructions:\n"
            "1. Pay close attention to the transcripts (`text`) of each clip. If the speaker mentions a sequence "
            "(e.g., 'first let's check out the trek, and once that is done I will give you the house tour' or "
            "'now let's move on to...', 'finally...'), you MUST order the clips to respect this spoken sequence.\n"
            "2. Group clips of the same topic together so the story flows naturally instead of jumping back and forth.\n"
            "3. Use the Context Document to understand the user's intent, the video theme, and any sequencing guidelines.\n\n"
            f"Context Document:\n{context_doc}\n\n"
            f"Clips to sequence:\n{json.dumps(middle)}\n\n"
            "You MUST output a valid JSON list of objects containing strictly these keys in the new sequenced order: "
            "'video_file', 'start', 'end', 'label', 'text', 'visual_description'.\n"
            "Do NOT add, remove, or modify the contents of any clip; only rearrange their order in the list.\n"
            "Return ONLY the JSON array inside a code block or as raw text."
        )
        
        logger.info(f"Requesting narrative sequencing from Gemini for {len(middle)} clips...")
        response = safe_generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        result_text = response.text.strip()
        sequenced_middle = json.loads(result_text)
        
        if isinstance(sequenced_middle, list):
            # Verify we didn't lose or corrupt any clips. If the count differs, fall back.
            if len(sequenced_middle) != len(middle):
                logger.warning(f"LLM returned {len(sequenced_middle)} clips but expected {len(middle)}. Falling back to default order.")
                return scenes
            
            # Map them back to the original dictionary items to keep any extra metadata
            mapped_middle = []
            for item in sequenced_middle:
                match = next(
                    (m for m in middle if m.get("video_file") == item.get("video_file") and abs(m.get("start", 0) - item.get("start", 0)) < 0.1),
                    None
                )
                if match:
                    mapped_middle.append(match)
                else:
                    # In case of minor rounding mismatch, use item directly
                    mapped_middle.append(item)
            
            logger.info("Successfully sequenced clips using Gemini narrative analysis.")
            return intros + mapped_middle + outros
        else:
            raise ValueError("Unexpected JSON format from LLM sequencing")
            
    except Exception as e:
        logger.error(f"LLM sequencing failed: {e}. Falling back to default order.")
        return scenes


def select_reel_segments_llm(scenes: List[Dict], target_duration: float, context_doc: str, user_prompt: str) -> Optional[List[Dict]]:
    """Use Gemini to select and sequence scenes for a short reel (target duration ~60s)."""
    if not init_gemini():
        return None
        
    try:
        from google.genai import types
        
        # Filter out FILLER segments first
        candidates = [s for s in scenes if s.get("label") != "FILLER"]
        
        # Target duration window: +/- 10s per minute
        min_dur = max(10.0, target_duration - (10.0 * (target_duration / 60.0)))
        max_dur = target_duration + (10.0 * (target_duration / 60.0))
        
        # We need to give the LLM clear instructions on duration budgeting and representation.
        prompt = (
            f"You are an expert video producer and editor. You are creating a short reel/vlog (target duration: {target_duration} seconds) "
            "from a list of raw video clips.\n\n"
            "Your task is to select a subset of clips that fit within the budget and sequence them "
            "to build an engaging, high-retention story.\n\n"
            "CRITICAL GUIDELINES:\n"
            f"1. DURATION BUDGET: The total duration of the selected clips MUST be strictly between {min_dur:.1f} and {max_dur:.1f} seconds. "
            "You must calculate the duration of each selected clip (end - start) and sum them up to fit this budget. This is a hard limit.\n"
            "2. FOCUS EVENT vs. DAY IN MY LIFE:\n"
            f"   - Check the User Prompt: '{user_prompt}'\n"
            "   - If the user specifies a specific event, speaker, or topic to focus on (e.g., 'focus on the house tour', 'highlight the trek portion'), "
            "then prioritize clips related to that event. You should build the core story around that event, but STILL include a few snippets "
            "from other video files surrounding the core event to support the narrative and show context.\n"
            "   - If the user does NOT specify a specific event (or if the prompt is empty or just generic like 'generate a vlog' or 'day in my life'), "
            "you MUST select at least one or more snippet(s) from EVERY SINGLE unique video file uploaded. This is crucial to build a balanced 'day in my life' montage.\n"
            "3. STORY & ENGAGEMENT: Arrange the selected clips in a logical, coherent, and highly engaging story flow. Use the transcripts and visual descriptions "
            "to create smooth transitions (e.g., placing intros first, outros last, and grouping similar topics together).\n"
            "4. PRESERVE NARRATIVE CUES & AVOID ABRUPT B-ROLLS:\n"
            "   - You MUST select and preserve talking clips (where the speaker is explaining the chronology, timeline, or context, e.g., 'first let's see some travel videos, and once that is done maybe then I will give you the house tour'). "
            "These segments are critical narrative anchors. Dropping them makes the video's progression confusing.\n"
            "   - Never include B-roll clips (such as scenery, flights, action) abruptly without including the corresponding narrative/talking scene that explains or sets up that context.\n"
            "   - Ensure you represent all uploaded files if they are key to the vlog's topics mentioned in the prompt (e.g. room tour and travel).\n"
            "5. BACKGROUND NOISE HANDLING:\n"
            "   - Deprioritize clips that are dominated by ambient background noise or public announcements (e.g. airport warnings, transit chimes, loud wind noise) so they are not treated as key narrative inputs, unless the user prompt/context explicitly requests to feature them. Do NOT completely remove these segments if they are part of the active vlog footage (e.g., walking outdoors); simply avoid prioritizing them or treating the noise as a significant narrative driver.\n\n"
            f"Context Document:\n{context_doc}\n\n"
            f"Available Candidates:\n{json.dumps(candidates)}\n\n"
            "You MUST output a valid JSON list of objects containing strictly these keys in the selected, sequenced order: "
            "'video_file', 'start', 'end', 'label', 'text', 'visual_description'.\n"
            "Do NOT modify the content of the selected clips (do not change timestamps, text, or filenames).\n"
            "Return ONLY the JSON array inside a code block or as raw text."
        )
        
        logger.info(f"Requesting reel selection and sequencing from Gemini for {len(candidates)} candidates...")
        response = safe_generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        result_text = response.text.strip()
        selected_scenes = json.loads(result_text)
        
        if isinstance(selected_scenes, list) and len(selected_scenes) > 0:
            # Map selected scenes back to original list items to preserve metadata
            mapped_scenes = []
            for item in selected_scenes:
                match = next(
                    (m for m in candidates if m.get("video_file") == item.get("video_file") and abs(m.get("start", 0) - item.get("start", 0)) < 0.1),
                    None
                )
                if match:
                    mapped_scenes.append(match)
                else:
                    mapped_scenes.append(item)
            logger.info(f"LLM successfully selected {len(mapped_scenes)} scenes for the reel.")
            return mapped_scenes
        else:
            raise ValueError("Unexpected JSON format from LLM reel selection")
            
    except Exception as e:
        logger.error(f"LLM reel selection failed: {e}. Falling back to Python heuristic selection.")
        return None

# ===========================================================================
# Phase 1 — EGT/EDL Schema Support
# ===========================================================================

def classify_egt_segments(segments: List[Dict], context_doc: str, progress_callback=None) -> List[Dict]:
    """Phase 1 Semantic Perception: Classify EGTSegment dicts using Gemini Flash Lite."""
    if not init_gemini():
        return segments  # Fallback to rule-based scores if no Gemini

    try:
        from google.genai import types

        class SegmentClassification(pydantic.BaseModel):
            segment_type: str = pydantic.Field(description="INTRO, OUTRO, SPEECH, B_ROLL, or SILENCE")
            structural_cue: Optional[str] = pydantic.Field(description="Any narrative transition spoken, e.g. 'let's go outside', 'before we start', or null")

        classified_segments = []
        total = len(segments)
        for i, seg in enumerate(segments):
            # Build 3-min rolling window for context
            rolling = build_rolling_window_summary(segments, i, window_sec=180.0)
            
            prompt = (
                "You are an expert video editor evaluating a single video segment.\n"
                "Classify its segment_type into exactly ONE of: INTRO, OUTRO, SPEECH, B_ROLL, or SILENCE.\n"
                "Also detect if the speaker is making a 'structural_cue' (e.g., 'let me show you the travel videos', 'now for the house tour').\n\n"
                f"Global Context:\n{context_doc}\n\n"
                f"Recent Transcript (rolling 3m):\n{rolling}\n\n"
                f"Target Segment to classify:\n{json.dumps({k:v for k,v in seg.items() if k in ['transcript', 'visual_description', 'start_sec', 'end_sec']})}"
            )

            try:
                response = safe_generate_content(
                    model="gemini-flash-lite-latest",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SegmentClassification,
                        temperature=0.1
                    )
                )
                data = json.loads(response.text.strip())
                # Update the segment dict
                seg["segment_type"] = data.get("segment_type", "SPEECH")
                seg["structural_cue"] = data.get("structural_cue")
                seg["perception_model"] = "gemini-flash-lite-latest"
            except pydantic.ValidationError as e:
                logger.warning(f"Failed to classify segment {seg.get('clip_id')}: {e}")
                # keep rule-based type fallback signaling
                seg["perception_model"] = "rule-based-v0"
            
            classified_segments.append(seg)

            # Fire progress callback after each segment so the UI stays alive
            if progress_callback:
                progress_callback(i + 1, total)
            
        return classified_segments
    except Exception as e:
        logger.error(f"Semantic classification failed: {e}")
        return segments


def generate_edl_llm(egt_json: Dict) -> Optional[List[Dict]]:
    """Phase 1 Reasoning: Generate a narrative, deduplicated EDL from the EGT."""
    if not init_gemini():
        logger.error("Gemini not initialized, cannot run Phase 1 Reasoning.")
        return None

    try:
        from google.genai import types

        class EDLEntrySchema(pydantic.BaseModel):
            clip_id: str = pydantic.Field(description="Must exactly match a clip_id from the EGT.")
            source_file: str = pydantic.Field(description="Must exactly match the source_file of the chosen clip_id.")
            start_sec: float = pydantic.Field(description="Start time from the EGT segment.")
            end_sec: float = pydantic.Field(description="End time from the EGT segment.")
            editorial_type: str = pydantic.Field(description="KEEP, INTRO, or OUTRO")
            sequence_index: int = pydantic.Field(description="0-indexed position in final timeline")

        class EDLSchema(pydantic.BaseModel):
            chain_of_thought: str = pydantic.Field(description="Think step-by-step about which clips are redundant/bad takes, and what structural cues imply reordering.")
            edl: List[EDLEntrySchema]

        prompt = (
            "You are the Director of 'VlogForge', an elite AI video editing system. "
            "Your task is to generate the final Edit Decision List (EDL) from the provided Editorial Ground Truth (EGT).\n\n"
            "VLOGFORGE CORE PHILOSOPHY & INSTRUCTIONS:\n"
            "1. GROUNDING CONSTRAINT: Every EDL entry MUST resolve to a real `clip_id` from the EGT. Do not hallucinate clips.\n"
            "2. CHRONOLOGY POLICY (Journey-structured): Default to chronological order. However, if there is an explicit structural cue "
            "(e.g., the speaker says 'let's look at the travel videos first'), you MUST break the chronological lock and reorder the clips to match the narrative intent.\n"
            "3. DEDUPLICATION & BAD TAKES: The EGT contains unedited raw takes. If the speaker stumbles, restarts a sentence, or repeats a phrase across multiple clips, "
            "you MUST DROP the redundant/bad takes (even if they have high quality scores) and ONLY KEEP the best, cleanest take to form a coherent sentence. Be ruthless with deduplication.\n"
            "4. MACRO-PACING: Ensure an INTRO is first (editorial_type: INTRO) and an OUTRO is last (editorial_type: OUTRO). All others are KEEP.\n"
            "5. NO SILENCE: Drop clips that are classified as SILENCE unless they are essential B-Roll.\n\n"
            f"=== EDITORIAL GROUND TRUTH (EGT) ===\n"
            f"{json.dumps(egt_json, indent=2)}\n\n"
            "Output the final EDL as a strictly validated JSON array according to the schema."
        )

        logger.info("Requesting Phase 1 Reasoning from gemini-2.5-flash...")
        response = safe_generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EDLSchema,
                temperature=0.2
            )
        )
        
        data = json.loads(response.text.strip())
        logger.info(f"Phase 1 Reasoning CoT: {data.get('chain_of_thought', '')}")
        return data.get("edl", [])
        
    except Exception as e:
        logger.error(f"Phase 1 Reasoning (generate_edl_llm) failed: {e}")
        return None


