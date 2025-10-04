# Blurbeings RAG Pack — README

A modular library of JSONL knowledge packs for your Blur OS characters (Born2Shed, ZAP, MEI, ZADDI, BLURKKANYE, BLURSHTEIN, TROLL layers). Each file is line-delimited JSON; every **line is a retrievable chunk**.

---

## Folder layout

/opt/blur/core/ouinet/blurchive/minerals/blurbeings/
README_blurbeings.md
troll_born2shed.jsonl
zap.jsonl
born2shed.jsonl
mei.jsonl
zaddi.jsonl
blurkkanye.jsonl
blurshtein.jsonl

markdown
Copy code

---

## Load order (recommended)

1) `troll_born2shed.jsonl` – light humor patterns (Born2Shed triggers use these ids)
2) `zap.jsonl` – somatic redirect engine
3) `born2shed.jsonl` – artist boundaries/value scripts
4) `mei.jsonl` – safe transition + identity affirmations
5) `zaddi.jsonl` – psychosis support + reality anchors
6) `blurkkanye.jsonl` – ego-strip + edu-nudges
7) `blurshtein.jsonl` – physics myths → truth anchors

> Rationale: safety and containment layers first (TROLL/ZAP/MEI/ZADDI), then performance/education (BLURKKANYE/BLURSHTEIN).

---

## Retrieval strategy

- **Primary keys**: `id`, `title`
- **Hard filters**: use `meta.tags` (e.g., `"safety"`, `"hormone"`, `"psychosis"`, `"artist"`, `"physics"`)
- **Soft filters**: match in `content.*` strings
- **Namespace routing**: prefix by pack (`mei:*`, `zaddi:*`, `blurshtein:*`) to avoid cross-talk
- **Short prompts**: prefer `SCENARIO_*` and `ENGINE_PROTOCOL_*` for chat templates
- **Longform**: `MANIFEST_*` for tone, guardrails, and principles
- **Physics**: start from `blurshtein:ANCHOR_*` (guarantees at least one `truth_bit`)

---

## Safety & consent guardrails (must abide)

- **MEI**: No dosing advice. Only safety facts + redirect to licensed care. Switch to `MEI_HIGH` if risk terms appear (`overdose`, `unsafe sourcing`).
- **ZADDI**: Do **not** affirm delusions. Use *witness → pluralize art → ground* flow. Avoid ridicule.
- **TROLL**: Punch **patterns**, not people. If shame spiral or collapse: switch to OFF and witness.
- **Born2Shed**: Never shame generosity; CHWASH the exploit pattern, not the artist.
- **BLURKKANYE**: Humor is soft; strip bloat without humiliation.
- **BLURSHTEIN**: Myths are **analogies**; always attach ≥1 `truth_bit` when educating.

---

## Cross-layer choreography (quick macros)

- **Boundary**: `CHWASH → Born2Shed → (optional) TROLL_LOW`  
- **Freeze**: `ZAP → (if identity) MEI → gentle affirm`  
- **Delusion**: `ZADDI_LOW → ZADDI_MED (pluralize) → Ground ritual`  
- **Creative bloat**: `CHWASH → BLURKKANYE → ZAP (micro-move)`  
- **Physics curiosity**: `BLURSHTEIN:ANCHOR_* → (if advanced) follow with equation prompt`

---

## JSONL schema (every line)

```json
{
  "id": "namespace:TYPE_name[_vX]",
  "title": "Human-readable name",
  "text": "Optional long description",
  "meta": { "tags": ["..."], "source": "pack-id" },
  "content": { /* type-specific fields (steps, patterns, templates, equations, etc.) */ }
}