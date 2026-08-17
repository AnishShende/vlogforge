# VlogForge Milestones

**M0 — Foundation Plumbing** *(needs building — retrofit)*
The scaffolding everything else sits on: auth, project CRUD, and resumable file upload (tus). Includes a platform-agnostic `ConnectedAccount` concept from day one — even though only YouTube is populated right now — so a future platform (Instagram) slots in without reworking the data model later.

**M1 — Tier 1 Perception** *(check completeness)*
The visual/audio tagging layer. Raw footage runs through Gemini to produce structured per-clip metadata — activity type, quality signals, notable moments — that every downstream stage depends on.

**M2 — First End-to-End Slice** *(check completness)*
Proof that the full pipeline holds together: Perception → Reasoning (LLM proposes an EDL) → Mechanical Assembly (deterministic repair/enforcement) → output. The goal here isn't polish, it's validating the architecture works end-to-end before scaling any single piece.

**M3 — Review UI** *(check completeness)*
The mandatory human-in-the-loop gate. The machine-proposed EDL is surfaced to the creator for approval or adjustment before final export — full automation was deliberately rejected in favor of this checkpoint.

**M4 — Long-Footage Scaling** *(next)*
Extending the pipeline from proof-of-concept clips to realistically long raw footage. Conceptually this is about handling scale limits gracefully — chunking/batching perception passes, and letting the reasoning tier work with much larger EDL candidate sets without losing narrative coherence.

**M5 — Metadata Generation** *(next — committed stopping point before the transit project)*
Auto-generating YouTube titles, descriptions, tags, hashtags, and captions. Tone and structural archetype are two independent selectors, not bundled. Candidates are presented as cards with tone/archetype labels visible, and grounding (why a suggestion was made) sits behind a small tappable info icon rather than an always-visible score.

**M6 — YouTube Integration + Vector DB**
Creators manually upload 4-5 of their own published videos (not pulled via API) to teach the system their "channel brand style" — treating the editing patterns in those references as house style, regardless of who actually edited them. A vector DB stores and retrieves style exemplars for personalization — chosen partly as a deliberate learning goal for this project, not pure necessity.

**M7 — Vertical-Specific Features**
Domain logic layered on top of the generic pipeline for the three target verticals — lifestyle, gym, travel. E.g. gym-specific priority handling for footage, vertical-specific pacing rules — built once the generic pipeline is solid rather than baked in early.

**M8 — Creator Profile Learning**
Closing the loop with quiet, ongoing personalization. The system learns a creator's recurring catchphrases and structural habits ("signature") in the background and improves suggestions over time — deliberately invisible, never surfaced as a profile the creator manages directly.

*Note: Build sequence is currently locked in through M5 before shifting focus to the transit pipeline project.*
