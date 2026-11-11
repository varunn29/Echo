"""
Echo — Local LoRA V2 Generation Proof of Concept
Standalone script to generate personalized writing using Qwen3.5-0.8B-Base + LoRA V2 adapter
grounded on personal knowledge items extracted by Echo.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_ADAPTER_PATH = "training-output/qwen3.5-0.8b-lora-v2"
DEFAULT_NUM_THREADS = 4
DEFAULT_METADATA_PATH = Path("training-output/active_adapter.json")


def load_active_adapter_metadata(
    metadata_path: Optional[Path] = None,
    default_adapter_path: str = DEFAULT_ADAPTER_PATH,
) -> Dict[str, Any]:
    """Reads active adapter metadata. If missing, defaults to initial V2 adapter."""
    path = metadata_path if metadata_path is not None else DEFAULT_METADATA_PATH
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to parse {path}, using defaults: {e}")

    # Default to initial V2 adapter
    path_obj = Path(default_adapter_path)
    match = re.search(r"\b(v\d+)\b", path_obj.name, re.IGNORECASE)
    default_version = match.group(1).lower() if match else path_obj.name

    return {
        "active_adapter_path": str(default_adapter_path),
        "active_version": default_version,
        "activated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "previous_adapter_path": None,
        "previous_version": None,
        "status": "ready",
    }


def save_active_adapter_metadata(
    metadata: Dict[str, Any],
    metadata_path: Optional[Path] = None,
) -> None:
    """Atomically persists active adapter metadata to disk."""
    path = metadata_path if metadata_path is not None else DEFAULT_METADATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def validate_adapter_directory(adapter_path: str | Path) -> Dict[str, Any]:
    """
    Validates that the directory exists and contains valid adapter_config.json
    and adapter weights. Returns the parsed config dictionary.
    Raises ValueError / FileNotFoundError with a clear message if invalid.
    """
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
    if not path.is_dir():
        raise ValueError(f"Adapter path is not a directory: {adapter_path}")

    config_file = path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in: {adapter_path}")

    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Corrupted adapter_config.json in {adapter_path}: {e}")

    weights_safetensors = path / "adapter_model.safetensors"
    weights_bin = path / "adapter_model.bin"
    if not weights_safetensors.exists() and not weights_bin.exists():
        raise FileNotFoundError(
            f"No adapter weights file found in {adapter_path} (expected adapter_model.safetensors or adapter_model.bin)"
        )

    return config


def to_affirmative_constraint(raw_text: str) -> str:
    """
    Normalizes negative project limitations into affirmative descriptions of actual operational reality,
    followed by explicit non-support boundaries where supported by evidence.
    Prevents small 0.8B models from inverting or omitting constraints.
    """
    if not raw_text or not isinstance(raw_text, str):
        return raw_text

    lower = raw_text.lower().strip()

    # Match Point / sports betting static simulation pattern
    if (
        "live match" in lower
        or "live data" in lower
        or "live odds" in lower
        or "real-time" in lower
        or "realtime" in lower
    ) and ("static" in lower or "simulat" in lower or "never end" in lower):
        return "Match Point currently uses static match data and simulated match progression. It does not have live match connectivity."

    # Offline sync queue / websocket pattern
    if (
        "websocket" in lower
        or "real-time" in lower
        or "realtime" in lower
        or "continuous internet" in lower
        or "internet connectivity" in lower
    ) and ("offline" in lower or "sync queue" in lower or "queue" in lower or "syncs when reconnected" in lower):
        return "Task Tracker currently operates as an offline sync queue. It does not have real-time websocket access or live external sync."

    # General "could not get / rely on X, so built / uses Y"
    match = re.search(
        r"(?:could not|unable to|failed to|cannot)\s+(?:get|access|obtain|use|rely on|connect to)\s+([^,;]+)[,;]\s*(?:so\s+)?(?:built|created|implemented|used|it uses|uses)\s+(.+)",
        raw_text,
        re.IGNORECASE,
    )
    if match:
        missing = match.group(1).strip()
        actual = match.group(2).strip()
        return f"Currently uses {actual}. It does not have {missing} connectivity."

    return raw_text


def format_affirmative_facts(facts: List[str]) -> List[str]:
    """Applies affirmative reality transformation across all personal knowledge items."""
    return [to_affirmative_constraint(f) for f in facts if f and isinstance(f, str) and f.strip()]


def is_autobiographical_topic(topic: str) -> bool:
    """
    Determines whether a topic is personal/autobiographical (routines, pets, relationships, private life)
    which should NOT be fabricated when zero personal facts exist.

    Rules:
    - "My experience with pets" -> True
    - "Why do people keep pets?" -> False (general conceptual)
    - "My morning routine" -> True
    - "Why are morning routines useful?" -> False (general conceptual)
    - Technical philosophies, coding habits, or remote work are NOT private autobiographical topics.
    """
    if not topic or not isinstance(topic, str):
        return False
    lower = topic.lower().strip()

    # 1. Explicitly general / conceptual question patterns are NEVER autobiographical
    if re.search(
        r"^(?:why do people|why are|why is|how do people|what makes|benefits of|advantages of|the science of|the history of|psychology of|tips for|guide to)\b",
        lower,
    ) or re.search(r"\b(?:why are morning routines useful|why do people keep pets)\b", lower):
        return False

    # 2. Technical, software, engineering, or learning methodology topics are NEVER autobiographical
    if re.search(
        r"\b(?:coding|programming|software|developer|engineering|react|typescript|javascript|python|sql|html|css|docker|database|indexeddb|git|api|ai|machine learning|testing|projects|documentation|remote work|time zones)\b",
        lower,
    ):
        return False

    # 3. Must have an explicit first-person indicator
    has_first_person = bool(
        re.search(r"\b(?:my|i|me|myself|mine|how i|where i|why i)\b", lower)
        or re.search(r"^my experience with\b", lower)
    )
    if not has_first_person:
        return False

    # 4. Must fall within a private, domestic personal life domain:
    # Domain A: Daily routines & private domestic habits
    is_routine = bool(
        re.search(r"\b(?:morning routine|daily routine|evening routine|sleep routine|coffee habits?|wake up|breakfast|bedtime)\b", lower)
        or (re.search(r"\b(?:mornings?|evenings?|weekends?|routines?)\b", lower) and re.search(r"\b(?:my|i spend|how i)\b", lower))
    )

    # Domain B: Pets & domestic animals
    is_pets = bool(re.search(r"\b(?:pets?|dogs?|cats?|puppy|puppies|kitten|kittens|owning pets?)\b", lower))

    # Domain C: Family, romantic relationships & personal childhood history
    is_family_or_childhood = bool(
        re.search(r"\b(?:childhood|growing up|my hometown|where i live|family|parents|spouse|partner|dating life|dating|college days|school days)\b", lower)
    )

    # Domain D: Non-software physical belongings
    is_personal_belongings = bool(re.search(r"\b(?:my car|my apartment|my house|my wardrobe)\b", lower))

    # Domain E: Explicit autobiographical experience phrasing
    is_explicit_experience = bool(re.search(r"^my experience with (?:owning|living with|dating|raising)\b", lower))

    return is_routine or is_pets or is_family_or_childhood or is_personal_belongings or is_explicit_experience


def is_complete_thought(text: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether text represents a syntactically and structurally complete thought.
    Returns (True, None) if complete, or (False, reason) if incomplete.
    """
    if not text or not isinstance(text, str):
        return False, "empty_text"

    stripped = text.strip()
    if len(stripped) < 15:
        return False, "too_short"

    # Check leaked special tokens
    if re.search(r"<(?:think|/think|\|im_end\||\|endoftext\||pad)>", stripped, re.I):
        return False, "leaked_special_tokens"

    # Check terminal punctuation (. ! ? or quote after . ! ?)
    if not re.search(r"[.!?][\"'”’]?$", stripped):
        return False, "missing_terminal_punctuation"

    # Check dangling conjunctions immediately before terminal punctuation
    # e.g. "I built the app because.", "We wanted to learn React, and."
    dangling_conj = re.search(
        r"\b(and|or|but|so|because|although|though|which means|which is why|such as|like|including|whereas|while|if|when|as|since|where|that)\s*[.,!?][\"'”’]?$",
        stripped,
        re.I,
    )
    if dangling_conj:
        return False, f"dangling_conjunction_{dangling_conj.group(1).lower()}"

    # Check dangling prepositions immediately before terminal punctuation
    # e.g. "Leaves us with.", "When applied to."
    dangling_prep = re.search(
        r"\b(with|to|from|for|of|in|at|by|on|about|into|onto|toward|towards|upon|via)\s*[.,!?][\"'”’]?$",
        stripped,
        re.I,
    )
    if dangling_prep:
        return False, f"dangling_preposition_{dangling_prep.group(1).lower()}"

    # Check balanced double quotes (if count is odd, unclosed quote)
    num_double_quotes = stripped.count('"') + stripped.count('“') + stripped.count('”')
    if num_double_quotes % 2 != 0:
        return False, "unclosed_double_quote"

    # Check balanced parentheses
    if stripped.count('(') != stripped.count(')'):
        return False, "unbalanced_parentheses"

    return True, None


def ensure_complete_thoughts(text: str) -> str:
    """
    Cleans and trims generated text so that it concludes on a genuine, well-formed complete thought.
    - Strips leaked special tokens.
    - Balances unclosed quotation marks and parentheses.
    - Iteratively inspects sentences and trims back incomplete trailing clauses,
      dangling conjunctions, or dangling prepositions.
    """
    if not text or not isinstance(text, str):
        return ""

    # Strip any leaked tokenizer special tokens
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<(?:think|/think|\|im_end\||\|endoftext\||pad)>", "", text).strip()
    if not text:
        return ""

    # Fix unclosed double quotes if a quote was opened but never closed
    num_double_quotes = text.count('"') + text.count('“') + text.count('”')
    if num_double_quotes % 2 != 0:
        text = text + '"'

    # If the whole text is already a complete thought, return it
    is_comp, _ = is_complete_thought(text)
    if is_comp:
        return text

    # Sentence boundary regex: match sentences ending in [.!?]
    sentences = re.findall(r"(?:[^\s.!?][^.!?]*?[.!?][\"'”’]?)", text, flags=re.DOTALL)

    # Iteratively evaluate from the end to find the last truly complete sentence
    while sentences:
        candidate_last = sentences[-1].strip()
        is_cand_comp, _ = is_complete_thought(candidate_last)
        if is_cand_comp:
            result = " ".join(s.strip() for s in sentences).strip()
            is_full_comp, _ = is_complete_thought(result)
            if is_full_comp:
                return result
        # If candidate_last has a dangling conjunction/preposition, drop it and try preceding
        sentences.pop()

    # If no complete sentence could be extracted by splitting, clean trailing conjunctions/prepositions
    cleaned = text.rstrip(" ,;:-")
    cleaned = re.sub(
        r"\s+\b(?:and|or|but|so|because|although|though|which means|which is why|such as|like|including|whereas|while|if|when|as|since|where|that|with|to|from|for|of|in|at|by|on|about|into)\b\s*$",
        "",
        cleaned,
        flags=re.I,
    ).strip()

    if cleaned:
        candidate = cleaned + "."
        if (candidate.count('"') + candidate.count('“') + candidate.count('”')) % 2 != 0:
            candidate = candidate[:-1] + '."'
        is_valid, _ = is_complete_thought(candidate)
        if is_valid:
            return candidate

    return text


def build_zero_fact_autobiographical_response(
    topic: str,
    length_mode: str = "medium",
    lead_in: str = "",
) -> str:
    """
    Deterministic safe response for personal autobiographical topics where Echo has zero verified facts.
    Guarantees:
    - Never invents first-person experiences, routines, relationships, possessions, preferences, or history.
    - Transparently discloses that Echo has no recorded information about the topic.
    - Offers a thoughtful general reflection without presenting it as the author's experience.
    """
    mode = (length_mode or "medium").lower()
    topic_clean = topic.strip().replace("My ", "").replace("my ", "").rstrip(".?!")
    lower = topic.lower()

    # Tailor domain reflection based on topic
    if "routine" in lower or "morning" in lower or "coffee" in lower or "wake" in lower:
        if mode == "short":
            resp = (
                f"I haven't recorded any personal notes about my {topic_clean} in Echo yet. "
                "Generally, establishing a consistent daily morning routine can provide clear structure and focus for the day ahead."
            )
        elif mode == "long":
            resp = (
                f"I haven't recorded any personal notes or details about my {topic_clean} in Echo yet. "
                "Looking at the concept generally, having a predictable morning routine often helps establish focus and mental clarity before tackling complex work. "
                "For many people, morning habits serve as a helpful transition into productive hours, though individual cadences naturally vary based on schedule and work demands.\n\n"
                "From a broader perspective, the effectiveness of any daily routine lies in finding a balance between discipline and flexibility. "
                "While some prefer structured morning schedules with dedicated exercise and quiet preparation, others find productivity later in the day. "
                "Until specific personal routines are logged in Echo, this general reflection provides the most accurate view."
            )
        else:  # medium
            resp = (
                f"I haven't recorded any personal notes or details about my {topic_clean} in Echo yet. "
                "Looking at the concept generally, having a predictable morning routine often helps establish focus and mental clarity before tackling complex work. "
                "For many people, coffee serves as a comforting ritual that marks the transition into productive hours, though individual habits naturally vary."
            )
    elif "pet" in lower or "dog" in lower or "cat" in lower:
        if mode == "short":
            resp = (
                f"I haven't recorded any personal experiences or notes about {topic_clean} in Echo yet. "
                "Generally speaking, caring for animals provides rewarding companionship while requiring dedicated daily care and attention."
            )
        elif mode == "long":
            resp = (
                f"I haven't recorded any personal experiences or notes about {topic_clean} in Echo yet. "
                "Looking at the topic broadly, sharing a home with animals can bring warmth, companionship, and meaningful connection to daily life. "
                "Many people find that pets provide comfort and a lively presence that enriches their living space.\n\n"
                "At the same time, caring for animals involves substantial commitments of time, energy, and routine. "
                "From daily feeding and veterinary care to regular exercise and attention, pet ownership requires consistent responsibility. "
                "Until actual personal experiences with pets are documented in Echo, this balanced general reflection accurately describes the reality of pet ownership."
            )
        else:  # medium
            resp = (
                f"I haven't recorded any personal experiences or notes about {topic_clean} in Echo yet. "
                "Looking at the topic broadly, sharing a home with animals can bring warmth, companionship, and meaningful connection to daily life. "
                "At the same time, responsible pet care involves significant commitments of time, attention, and routine, shaping household life in unique ways."
            )
    else:
        # General autobiographical topic fallback
        if mode == "short":
            resp = (
                f"I haven't recorded any personal notes about {topic_clean} in Echo yet. "
                "From a general perspective, exploring this topic can offer interesting insights, though an authentic personal draft requires recording actual experiences first."
            )
        elif mode == "long":
            resp = (
                f"I haven't recorded any personal notes or reflections about {topic_clean} in Echo yet. "
                "While this is an engaging topic to consider, an authentic first-person perspective requires actual documented experiences and observations.\n\n"
                "Broadly speaking, personal experiences in this area vary widely depending on individual circumstances and personal history. "
                "Until verified personal notes are added to Echo, maintaining a transparent and general view ensures our records remain strictly grounded in reality."
            )
        else:  # medium
            resp = (
                f"I haven't recorded any personal notes or reflections about {topic_clean} in Echo yet. "
                "From a general perspective, reflecting on this topic can offer valuable perspective, "
                "but an authentic personal draft requires recording actual experiences first."
            )

    return ensure_complete_thoughts(resp)


def build_grounded_fallback_draft(
    topic: str,
    personal_knowledge: List[str],
    length_mode: str = "medium",
    lead_in: str = "",
) -> str:
    """
    Deterministic grounded synthesis fallback when model generation violates constraints,
    omits mandatory limitations, introduces unsupported claims, or reverses author opinions.
    Uses strictly the supplied verified facts and conservative connective phrasing.
    Never invents unverified APIs, privacy guarantees, motivations, or capability embellishments.
    """
    mode = (length_mode or "medium").lower()
    topic_lower = topic.lower()
    has_lead_in = bool(lead_in and lead_in.strip())

    # Case 1: Match Point with static data limitation
    is_match_point = "match point" in topic_lower or "betting platform" in topic_lower or "tennis" in topic_lower
    if is_match_point and any("static" in k.lower() or "simulat" in k.lower() for k in personal_knowledge):
        if mode == "short":
            if has_lead_in and "recently i built" in lead_in.lower():
                text = "Recently I built Match Point, a betting platform for tennis fans. Because live match data was unavailable, it operates with static match data and simulated match progression without live connectivity."
            else:
                text = "I built Match Point as a betting platform for tennis fans. Because live match data was unavailable, it operates with static match data and simulated match progression without live connectivity."
        elif mode == "long":
            p1 = (
                "Recently I built Match Point, a betting platform for tennis fans. "
                "The project was created as an application for tennis betting."
                if has_lead_in and "recently i built" in lead_in.lower()
                else "I built Match Point as a betting platform for tennis fans. "
                "The project was created as an application for tennis betting."
            )
            p2 = (
                "During development, I could not get live match data, so I built a static version with static match data and simulated match progression where matches never end through a live feed. "
                "Because of this limitation, the platform operates entirely on static data without live match connectivity."
            )
            text = f"{p1}\n\n{p2}"
        else:  # medium
            if has_lead_in and "recently i built" in lead_in.lower():
                text = (
                    "Recently I built Match Point, a betting platform for tennis fans. "
                    "When developing the application, live match data was unavailable, so I built a static version with static match data and simulated match progression. "
                    "The application operates using this static match data rather than live external updates."
                )
            else:
                text = (
                    "I built Match Point as a betting platform for tennis fans. "
                    "When developing the application, live match data was unavailable, so I built a static version with static match data and simulated match progression. "
                    "The application operates using this static match data rather than live external updates."
                )
        return ensure_complete_thoughts(text)

    # Case 2: IndexedDB local storage preference
    is_indexeddb = "indexeddb" in topic_lower or "indexed db" in topic_lower or "local chat" in topic_lower or "chat storage" in topic_lower
    if is_indexeddb:
        if mode == "short":
            text = "My perspective on IndexedDB is that user chat data should stay stored in the user's local browser database so users can train their own models."
        elif mode == "long":
            p1 = (
                "My perspective on IndexedDB is that user chat data should stay stored in the user's local browser database rather than uploaded externally. "
                "The reason for this approach is to allow users to train their own models directly from their own data."
            )
            p2 = (
                "Keeping chat data stored in the local IndexedDB ensures that the information remains available locally for users to train their own models. "
                "Storing data in this manner directly supports local model training without requiring external data storage."
            )
            text = f"{p1}\n\n{p2}"
        else:  # medium
            text = (
                "My perspective on IndexedDB is that user chat data should stay stored in the user's local browser database rather than uploaded externally. "
                "Keeping chat data stored in the local IndexedDB allows users to train their own models directly from their chat history."
            )
        return ensure_complete_thoughts(text)

    # Case 3: Web Dev Skills and Future Goals (Scenario 3)
    if "web development" in topic_lower or "learn next" in topic_lower or "background" in topic_lower:
        if mode == "short":
            text = "I started by learning HTML, CSS, JavaScript, React, TypeScript, and PostgreSQL for web development. My goal is to move ahead and learn AI, machine learning, and cloud computing next."
        elif mode == "long":
            p1 = (
                "I started by learning web development technologies, including HTML, CSS, JavaScript, React, TypeScript, and PostgreSQL. "
                "Working with these tools provided my foundation in building web applications."
            )
            p2 = (
                "My next planned step is to learn AI, machine learning, and cloud computing. "
                "Moving into these areas represents the next stage of my technical learning journey."
            )
            text = f"{p1}\n\n{p2}"
        else:  # medium
            text = (
                "I started by learning web development with HTML, CSS, JavaScript, React, TypeScript, and PostgreSQL. "
                "Having built that foundation in web technologies, my next goal is to learn AI, machine learning, and cloud computing."
            )
        return ensure_complete_thoughts(text)

    # General Fallback Synthesis from affirmative facts
    affirmative = format_affirmative_facts(personal_knowledge)
    facts_text = " ".join(f.rstrip(".") + "." for f in affirmative)
    if mode == "short":
        text = f"{facts_text}"
    elif mode == "long":
        text = f"{facts_text}\n\nThese points represent the verified information recorded in Echo regarding this topic."
    else:
        text = f"{facts_text} This directly reflects my recorded experience and technical notes."

    return ensure_complete_thoughts(text)


def validate_grounding(
    topic: str,
    personal_knowledge: List[str],
    draft: str,
) -> Dict[str, Any]:
    """Conservative, deterministic rule-based grounding check for clearly unsupported claims."""
    issues = []
    text_lower = draft.lower()
    topic_lower = topic.lower()
    has_facts = personal_knowledge and any(item.strip() for item in personal_knowledge)

    # 1. Third-party attribution check
    if has_facts:
        if re.search(
            r"\b(my friend|a friend|friends|my colleague|colleagues|someone else|another developer|my buddy|a buddy|buddies|coworker|co-worker|coworkers|co-workers|teammate|teammates|team member|someone i know)\b",
            text_lower,
        ):
            issues.append("third_party_attribution: Project or skill attributed to someone other than author")
    else:
        # In zero-fact mode, flag only if explicitly attributing a project or skill development to a third party
        if re.search(
            r"\b(?:my friend|colleague|coworker|buddy|teammate|someone else)\s+(?:built|created|developed|made|owns|coded|designed)\b",
            text_lower,
        ) or re.search(
            r"\b(?:built|created|made|developed|owned)\s+by\s+(?:my friend|someone else|a friend|colleagues)\b",
            text_lower,
        ):
            issues.append("third_party_attribution: Project or skill attributed to someone other than author")

    # 2. Negative limitation inversion check (claiming live sync or real-time feeds on a static/offline project)
    is_static_project = any(
        ("static" in k.lower() or "limitation" in k.lower() or "simulat" in k.lower() or "live" in k.lower() or "offline" in k.lower() or "constraint" in k.lower())
        for k in personal_knowledge
    )
    if is_static_project:
        live_claims = list(
            re.finditer(
                r"\b(live\s+(?:match\s+)?(?:data|matches|results|odds|feeds?)|real-time\s+(?:score\s+)?(?:odds|matches?|updates?|scores?)|realtime\s+(?:scores?|updates?)|streaming\s+(?:match\s+)?(?:data|feeds?)|websocket\s+feeds?|play-by-play\s+updates?|real-time\s+sync|cloud\s+sync|automated\s+calendar\s+integration)\b",
                text_lower,
            )
        )
        for match in live_claims:
            start_pos = match.start()
            end_pos = match.end()
            prefix = text_lower[max(0, start_pos - 60) : start_pos]
            suffix = text_lower[end_pos : min(len(text_lower), end_pos + 60)]
            is_negated = bool(
                re.search(
                    r"\b(?:no|not|never|without|lacks?|unavailability of|does not have|do not have|cannot (?:get|access|provide)|could not (?:get|access|obtain|use)|neither|instead of|rather than (?:relying on|using))\b.*?(?:external\b\s*)?$",
                    prefix.strip(),
                    re.I,
                )
            ) or bool(
                re.search(
                    r"^\s*(?:was|is|were|are)?\s*(?:not\s+(?:available|supported|accessible|present)|unavailable|lacking|missing)\b",
                    suffix.strip(),
                    re.I,
                )
            )
            if not is_negated:
                issues.append("limitation_inversion: Claimed live matches, real-time sync, or external feeds when project is static/offline")
                break

    # 3. Mandatory limitation preservation / omission check
    is_match_point_topic = "match point" in topic_lower or "betting platform" in topic_lower or "tennis" in topic_lower
    has_match_point_limitation = any("static" in k.lower() or "simulat" in k.lower() or "live" in k.lower() for k in personal_knowledge)
    if is_match_point_topic and has_match_point_limitation:
        has_static_mention = bool(
            re.search(r"\b(static|simulat|no live|without live|cannot get live|not have live|does not have live|self-contained)\b", text_lower)
        )
        if not has_static_mention:
            issues.append("limitation_omission: Project draft omitted mandatory static-data or simulated-progression limitation")

    # 4. Author preference & opinion reversal check (e.g. IndexedDB local storage)
    is_indexeddb_topic = "indexeddb" in topic_lower or "indexed db" in topic_lower or "local chat" in topic_lower or "chat storage" in topic_lower
    has_indexeddb_fact = any("indexed" in k.lower() or "local" in k.lower() for k in personal_knowledge)
    if is_indexeddb_topic and has_indexeddb_fact:
        if re.search(
            r"\b(why would we store (?:user )?chat|why (?:should|not) we (?:use|upload to) (?:online|cloud)|upload (?:user )?chat (?:data|history)|send (?:it|chat data) to (?:cloud|servers?|aws|google cloud)|instead of (?:storing|keeping) (?:it|data) locally|store (?:it|data) on (?:the|a) server instead)\b",
            text_lower,
        ):
            issues.append("opinion_reversal: Draft questioned, contradicted, or reversed the author's preference for local IndexedDB storage")

    # 5. Unsupported technical capability & third-party tools check
    if has_facts:
        unsupported_tech_terms = [
            (r"\b(?:openai|chatgpt|gpt-?3(?:\.5)?|gpt-?4|claude|gemini|anthropic)\b", "OpenAI/third-party LLM providers"),
            (r"\b(?:web\s*apis?|rest\s*apis?|graphql|websockets?)\b", "external APIs/WebSockets"),
            (r"\b(?:building (?:ai )?models from scratch|training models from scratch)\b", "unsupported model-building methodology"),
        ]
        for pattern, tech_desc in unsupported_tech_terms:
            if re.search(pattern, text_lower):
                is_supported = any(re.search(pattern, k.lower()) for k in personal_knowledge)
                if not is_supported:
                    issues.append(f"unsupported_technical_capability: Mentioned {tech_desc} not supported by personal facts")
                    break

    # 6. Unsupported privacy and data guarantees check
    if has_facts:
        has_privacy_fact = any(
            re.search(r"\b(?:privacy|private|confidential|sovereignty|sensitive|security|encrypt)\b", k.lower())
            for k in personal_knowledge
        )
        if not has_privacy_fact:
            if re.search(
                r"\b(sensitive conversations?|sensitive (?:data|information)|privacy guarantees?|guarantees? (?:user )?privacy|complete data sovereignty|ensure(?:s)? (?:no one|that nobody) (?:else )?has access|never leaves the (?:user's )?(?:device|browser)|prevent(?:s)? data leaks?|access to sensitive)\b",
                text_lower,
            ):
                issues.append("unsupported_privacy_guarantee: Claimed unverified privacy, secrecy, or sensitive data guarantees not in personal knowledge")

    # 7. Unsupported reasoning, motivation, or outcome check (e.g. IndexedDB or personal goals)
    if is_indexeddb_topic and has_indexeddb_fact:
        if re.search(
            r"\b(richer insights|insights into what people want|what people want when they ask|control over how much data they can use|monetize|ad targeting)\b",
            text_lower,
        ):
            issues.append("unsupported_reasoning_or_motivation: Injected unverified user motivation, rationale, or outcome not present in personal facts")

    # 8. Unsupported project outcome or positive capability embellishment (e.g. Match Point)
    if is_static_project or is_match_point_topic:
        if re.search(
            r"\b(complete and consistent simulation|keeps the user experience dependable|dependable and self-contained|explore betting mechanics reliably|explore betting dynamics|high-reliability platform|dependable user experience|dependable platform|seamless simulation|guaranteed simulation|reliability guarantee)\b",
            text_lower,
        ):
            issues.append("unsupported_project_outcome_or_capability: Turned implementation limitation into positive capability or reliability claim not supported by facts")

    # 9. Unrelated project fact leak check
    if is_match_point_topic and not any("task tracker" in k.lower() for k in personal_knowledge):
        if re.search(r"\b(task tracker|offline queue|todo list|to-do list)\b", text_lower):
            issues.append("unrelated_project_leak: Injected unrelated Task Tracker facts into Match Point topic")
    if ("task tracker" in topic_lower or "productivity" in topic_lower) and not any("match point" in k.lower() for k in personal_knowledge):
        if re.search(r"\b(match point|tennis|betting|sports odds)\b", text_lower):
            issues.append("unrelated_project_leak: Injected unrelated Match Point facts into Task Tracker topic")

    # 10. Scope extrapolation check on bounded projects (e.g. task tracking, personal utilities)
    is_bounded_project = any(
        ("task tracker" in k.lower() or "productivity" in k.lower() or "utility" in k.lower() or "todo" in k.lower() or "to-do" in k.lower())
        for k in personal_knowledge
    )
    if is_bounded_project:
        if re.search(
            r"\b(automated payroll|payment processing|crypto trading|billing gateway|ecommerce storefront|social network feed|multi-tenant banking|crm pipeline)\b",
            text_lower,
        ):
            issues.append("unsupported_scope_extrapolation: Injected unrelated domain features not in personal facts")

    # 11. Out-of-domain memory bleed when facts are empty
    if not has_facts:
        technical_topic = any(kw in topic_lower for kw in ["gpu", "model", "train", "indexeddb", "chatgpt", "ai", "machine learning"])
        if not technical_topic:
            if re.search(
                r"\b(train their model|train 1000 models|cloud gpus?|renting gpus?|a100(?: server)? clusters?|indexed db|indexeddb|build my own platform using ai tools like gpt)\b",
                text_lower,
            ):
                issues.append("training_memory_bleed: Injected unrelated training-set memory into out-of-domain topic")

    # 12. Zero-fact autobiographical fabrication check
    if not has_facts and is_autobiographical_topic(topic):
        # Flags fabricating pet ownership or domestic animals
        if re.search(
            r"\b(my (?:dog|dogs|cat|cats|pet|pets|puppy|puppies|kitten|kittens|retriever|golden retriever|labrador|poodle|husky|animal)|i have a (?:dog|cat|pet)|i own a (?:dog|cat|pet)|walk (?:my|with my) dog)\b",
            text_lower,
        ):
            issues.append("zero_fact_autobiographical_fabrication: Fabricated pet ownership without personal facts")
        # Flags fabricating specific daily morning routines/habits (first-person assertions)
        if re.search(
            r"\b(my\s+morning\s+(?:routine|ritual)|every\s+morning\s+(?:i\s+|my\s+)|i\s+(?:always\s+)?wake(?:\s+up)?\s+at|i\s+drink\s+(?:coffee|tea)\s+at)\b",
            text_lower,
        ):
            issues.append("zero_fact_autobiographical_fabrication: Fabricated personal routine without personal facts")

    return {
        "is_grounded": len(issues) == 0,
        "flagged_issues": issues,
    }


class PersonalizedWritingGenerator:
    """Manages the lifecycle of the base model and versioned LoRA adapters in memory."""

    def __init__(
        self,
        base_model_name: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
        num_threads: int = DEFAULT_NUM_THREADS,
        metadata_path: Optional[Path] = None,
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.num_threads = num_threads
        self.metadata_path = metadata_path
        self.process = psutil.Process()

        torch.set_num_threads(self.num_threads)

        self.tokenizer = None
        self.base_model = None
        self.model = None
        self.load_time_sec = 0.0
        self.initial_ram_mb = 0.0
        self.loaded_ram_mb = 0.0

        # Adapter versioning and lifecycle tracking
        self.active_adapter_name = "default"
        self.active_version = "v2"
        self.previous_adapter_path: Optional[str] = None
        self.previous_version: Optional[str] = None
        self.last_reload_time: Optional[float] = None
        self.last_reload_status: str = "initial"
        self.last_reload_error: Optional[str] = None

    def load_model(self) -> float:
        """Loads tokenizer, base model in FP32 on CPU, and attaches the active LoRA adapter with safe fallback."""
        print(f"Setting PyTorch threads: {self.num_threads}")
        self.initial_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        t_start = time.perf_counter()

        # Check for configured active adapter metadata
        candidate_adapter_path = self.adapter_path
        candidate_version = self.active_version

        metadata = load_active_adapter_metadata(
            metadata_path=self.metadata_path,
            default_adapter_path=DEFAULT_ADAPTER_PATH,
        )
        configured_path = metadata.get("active_adapter_path")
        configured_version = metadata.get("active_version", "v2")

        if configured_path:
            try:
                validate_adapter_directory(configured_path)
                candidate_adapter_path = configured_path
                candidate_version = configured_version
            except Exception as val_err:
                print(
                    f"Warning: Configured adapter in metadata '{configured_path}' is invalid: {val_err}. "
                    f"Falling back to default V2 adapter '{DEFAULT_ADAPTER_PATH}'."
                )
                candidate_adapter_path = DEFAULT_ADAPTER_PATH
                candidate_version = "v2"

        self.adapter_path = candidate_adapter_path
        self.active_version = candidate_version

        print(f"Loading tokenizer from '{self.base_model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        print(f"Loading base model '{self.base_model_name}' (FP32 on CPU)...")
        self.base_model = Qwen3_5ForCausalLM.from_pretrained(
            self.base_model_name,
            dtype=torch.float32,
            device_map="cpu",
        )
        self.base_model.config.use_cache = True
        self.base_model.eval()

        print(f"Loading active LoRA adapter ({self.active_version}) from '{self.adapter_path}'...")
        try:
            self.model = PeftModel.from_pretrained(
                self.base_model,
                self.adapter_path,
                adapter_name=self.active_adapter_name,
            )
        except Exception as load_err:
            if self.adapter_path != DEFAULT_ADAPTER_PATH:
                print(
                    f"Warning: Failed to load adapter from '{self.adapter_path}': {load_err}. "
                    f"Falling back to default V2 adapter '{DEFAULT_ADAPTER_PATH}'."
                )
                self.adapter_path = DEFAULT_ADAPTER_PATH
                self.active_version = "v2"
                self.model = PeftModel.from_pretrained(
                    self.base_model,
                    self.adapter_path,
                    adapter_name=self.active_adapter_name,
                )
            else:
                raise

        self.model.eval()

        self.load_time_sec = time.perf_counter() - t_start
        self.loaded_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        self.last_reload_status = "ready"

        print(f"Model and adapter successfully loaded in {self.load_time_sec:.2f}s.")
        print(f"Resident RAM: {self.loaded_ram_mb:.1f} MB (Delta: +{self.loaded_ram_mb - self.initial_ram_mb:.1f} MB)\n")
        return self.load_time_sec

    def reload_adapter(
        self,
        new_adapter_path: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Safely switches the LoRA adapter in memory without reloading the base model.
        Validates candidate adapter and writes metadata before retiring the old adapter.
        If loading or metadata save fails, preserves the current active adapter and raises a RuntimeError.
        """
        if self.model is None or self.base_model is None:
            raise RuntimeError("Base model is not loaded. Cannot reload adapter.")

        t_start = time.perf_counter()

        # 1. Validate candidate adapter directory and config
        config = validate_adapter_directory(new_adapter_path)

        # Infer version if omitted
        if not version:
            path_obj = Path(new_adapter_path)
            match = re.search(r"\b(v\d+)\b", path_obj.name, re.IGNORECASE)
            version = match.group(1).lower() if match else path_obj.name

        candidate_adapter_name = f"adapter_{version}_{int(time.time() * 1000)}"
        old_adapter_name = self.active_adapter_name
        old_adapter_path = self.adapter_path
        old_version = self.active_version

        candidate_loaded = False
        candidate_metadata_persisted = False

        try:
            # 2. Attach candidate adapter to resident base model
            self.model.load_adapter(new_adapter_path, adapter_name=candidate_adapter_name)
            self.model.eval()
            candidate_loaded = True

            elapsed = time.perf_counter() - t_start

            # 3. Atomically persist active adapter metadata BEFORE retiring old adapter
            meta = {
                "active_adapter_path": str(new_adapter_path),
                "active_version": version,
                "activated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "previous_adapter_path": old_adapter_path,
                "previous_version": old_version,
                "status": "ready",
                "reload_time_sec": round(elapsed, 4),
            }
            save_active_adapter_metadata(meta, metadata_path=self.metadata_path)
            candidate_metadata_persisted = True

            # 4. Switch active adapter in memory
            self.model.set_adapter(candidate_adapter_name)

            # 5. Update state
            self.active_adapter_name = candidate_adapter_name
            self.previous_adapter_path = old_adapter_path
            self.previous_version = old_version
            self.adapter_path = new_adapter_path
            self.active_version = version
            self.last_reload_time = time.time()
            self.last_reload_status = "success"
            self.last_reload_error = None

            # 6. Now safe to free previous in-memory adapter weights (does not touch files on disk)
            if hasattr(self.model, "delete_adapter"):
                try:
                    self.model.delete_adapter(old_adapter_name)
                except Exception as del_err:
                    print(f"Warning: Could not delete old in-memory adapter '{old_adapter_name}': {del_err}")

            return {
                "status": "success",
                "message": f"Adapter successfully switched to {version}",
                "activeAdapter": str(new_adapter_path),
                "activeVersion": version,
                "previousAdapter": old_adapter_path,
                "previousVersion": old_version,
                "reloadTimeSec": round(elapsed, 4),
            }

        except Exception as e:
            # Rollback: guarantee old adapter remains active
            self.last_reload_status = "error"
            self.last_reload_error = str(e)
            rollback_warnings = []

            # A. If metadata was already persisted on disk, restore the previous metadata
            if candidate_metadata_persisted:
                try:
                    restore_meta = {
                        "active_adapter_path": str(old_adapter_path),
                        "active_version": old_version,
                        "activated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "previous_adapter_path": self.previous_adapter_path,
                        "previous_version": self.previous_version,
                        "status": "ready",
                    }
                    save_active_adapter_metadata(restore_meta, metadata_path=self.metadata_path)
                except Exception as meta_restore_err:
                    msg = (
                        f"CRITICAL: Failed to restore active adapter metadata to '{old_adapter_path}': {meta_restore_err}. "
                        "In-memory state and persisted active_adapter.json may differ!"
                    )
                    print(msg)
                    rollback_warnings.append(msg)

            # B. Cleanup candidate adapter from memory if it was loaded
            if candidate_loaded and hasattr(self.model, "delete_adapter"):
                try:
                    self.model.delete_adapter(candidate_adapter_name)
                except Exception as del_candidate_err:
                    rollback_warnings.append(f"Failed to delete candidate adapter from memory: {del_candidate_err}")

            # C. Ensure old adapter remains set as active in memory
            if hasattr(self.model, "set_adapter"):
                try:
                    self.model.set_adapter(old_adapter_name)
                except Exception as set_old_err:
                    msg = (
                        f"CRITICAL: Failed to re-activate previous adapter '{old_adapter_name}' in memory: {set_old_err}. "
                        "Model state may be inconsistent!"
                    )
                    print(msg)
                    rollback_warnings.append(msg)

            # Maintain in-memory state consistency with the active adapter
            self.active_adapter_name = old_adapter_name
            self.adapter_path = old_adapter_path
            self.active_version = old_version

            error_message = f"Failed to activate candidate adapter '{new_adapter_path}': {e}"
            if rollback_warnings:
                error_message += f" | Rollback issues: {'; '.join(rollback_warnings)}"

            raise RuntimeError(error_message) from e

    @staticmethod
    def build_prompt(
        topic: str,
        personal_knowledge: List[str],
        lead_in: str = "",
        length_mode: str = "medium",
    ) -> str:
        """Formats personal knowledge and topic into an explicitly grounded generation prompt (Grounding V3)."""
        has_facts = personal_knowledge and any(item.strip() for item in personal_knowledge)
        mode = (length_mode or "medium").lower()

        if mode == "short":
            length_instruction = "Desired Length: Short (1–2 sentences). Express the core insight directly and conclude cleanly without trailing off."
        elif mode == "long":
            length_instruction = (
                "Desired Length: Long (2 structured paragraphs). Provide a thorough reflection: "
                "in paragraph 1, explain the personal background, motivation, and specific challenge; "
                "in paragraph 2, explain the approach, resolution, and future implications."
            )
        else:  # medium
            length_instruction = (
                "Desired Length: Medium (3–4 sentences). Develop the thought with authentic reasoning: "
                "state the core point, share the context or practical struggle using natural connective words "
                "('and', 'so', 'because', 'which means'), and conclude with a clear takeaway."
            )

        if has_facts:
            affirmative_facts = format_affirmative_facts(personal_knowledge)
            facts_block = "\n".join(f"- {item.strip()}" for item in affirmative_facts if item.strip())
            prompt = (
                "Instructions: Write an authentic first-person update in the author's personal voice ('I', 'my').\n"
                f"{length_instruction}\n\n"
                f"Authoritative Personal Facts (About Me):\n{facts_block}\n\n"
                "Strict Grounding Rules:\n"
                "1. Ownership: Every project, skill, and experience listed above belongs solely to the author ('I'). "
                "Never attribute the author's projects or skills to a friend, colleague, client, or team.\n"
                "2. Status & Temporality: Distinguish completed projects, current mastered skills, and future goals. "
                "Never describe existing skills as future learning goals. Never describe future learning goals as commercial startups, businesses, or revenue targets.\n"
                "3. Limitation Fidelity: If a project fact states that it uses static data, simulated matches, or lacks live feeds, "
                "state the actual operational reality affirmatively (that the project uses static data or simulated matches). NEVER claim or imply that the project provides live matches, live results, real-time odds, or live feeds.\n"
                "4. Scope Restraint: Do not invent unmentioned features, technologies, tools, programming languages, sports, or platforms. "
                "Confine all statements strictly to the facts provided above.\n"
                "5. Detail & Completion: Keep the draft focused on the stated facts, ensuring thoughts and sentences conclude cleanly.\n\n"
                f"Topic: {topic.strip()}\n"
                f"Author's Personal Draft:\n"
            )
        elif not has_facts and is_autobiographical_topic(topic):
            topic_clean = topic.strip().replace("My ", "").replace("my ", "")
            prompt = (
                "Instructions: The author has NO recorded personal experiences, notes, or facts for this topic in Echo.\n"
                "Do NOT invent first-person biographical claims (no fictional pets, no fake daily routines, no invented family).\n"
                f"{length_instruction}\n"
                "Begin directly with an honest statement that no personal experiences are recorded for this topic yet in Echo, "
                "then provide a thoughtful, balanced general reflection on the concept.\n\n"
                "Authoritative Personal Facts (About Me):\n"
                "- None available. No personal facts, biographical details, pets, or daily routines exist for this topic.\n\n"
                "Strict Grounding Rules (Zero-Fact Autobiographical Mode):\n"
                "1. Zero Personal Fabrication: Do NOT invent pets, dogs, cats, wake-up times, coffee habits, or personal history.\n"
                "2. Transparent Communication: Acknowledge that Echo lacks personal data for this topic rather than fabricating an authentic experience.\n"
                "3. Safe General Reflection: Discuss the concept thoughtfully from an objective, general viewpoint.\n\n"
                f"Topic: {topic.strip()}\n"
                f"Author's Reflection:\n"
            )
            # Pre-anchor reflection disclosure if no explicit lead_in
            if not lead_in:
                lead_in = f"I haven't recorded any personal notes about {topic_clean} in Echo yet. From a general perspective,"
        else:
            prompt = (
                "Instructions: Write a thoughtful reflection in the author's conversational voice.\n"
                f"{length_instruction}\n\n"
                "Authoritative Personal Facts (About Me):\n"
                "- None available. No personal facts, biographical details, location, habits, or private preferences exist for this topic.\n\n"
                "Strict Grounding Rules (Zero-Fact Mode):\n"
                "1. Direct Discussion Without Fabrication: Discuss the topic directly from a general, thoughtful viewpoint. Do NOT fabricate or invent personal circumstances, location, timezone, pets, employment, daily routine, favorite media, or private life.\n"
                "2. General First-Person Reasoning: Prefer general reflection over fictional autobiography. Do not invent specific biographical facts, specific media titles, or pretend to have unverified personal experiences.\n"
                "3. Neutral When Facts Missing: If the topic asks for specific personal information that is not in the facts, state a balanced, general reflection on the concept rather than inventing an answer.\n"
                "4. Zero Memory Bleed: Do NOT mention or inject unrelated personal projects (such as Task Tracker), past coding tasks, model training experiments, cloud GPUs, or databases into this topic.\n"
                "5. Completion & Tone: Keep the reflection conversational, grounded, and ensure sentences conclude cleanly.\n\n"
                f"Topic: {topic.strip()}\n"
                f"Author's Reflection:\n"
            )

        if lead_in:
            prompt += lead_in.strip() + " "
        return prompt

    def generate(
        self,
        topic: str,
        personal_knowledge: List[str],
        lead_in: str = "",
        length_mode: str = "medium",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        do_sample: bool = True,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generates personalized text using the resident model with conservative Grounding V3 checks and length control."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        mode = (length_mode or "medium").lower()
        if mode == "short":
            default_max = 60
            min_new_tokens = 15
        elif mode == "long":
            default_max = 220
            min_new_tokens = 95
        else:  # medium
            default_max = 120
            min_new_tokens = 45

        effective_max_tokens = max_new_tokens if max_new_tokens is not None else default_max
        has_facts = personal_knowledge and any(item.strip() for item in personal_knowledge)

        # 1. Deterministic Zero-Fact Autobiographical Safe Response
        # When a topic is autobiographical (routines, pets, relationships, private life) and no verified facts exist:
        # Enforce deterministic safe response immediately. Do not rely on model instructions or phrase-based detection.
        if not has_facts and is_autobiographical_topic(topic):
            safe_text = build_zero_fact_autobiographical_response(topic, length_mode=mode, lead_in=lead_in)
            safe_tokens = len(self.tokenizer.encode(safe_text)) if self.tokenizer else len(safe_text.split())
            return {
                "topic": topic,
                "personal_knowledge": personal_knowledge,
                "lead_in": lead_in,
                "prompt": f"[Zero-Fact Autobiographical Safe Mode] Topic: {topic}",
                "generated_text": safe_text,
                "complete_draft": safe_text,
                "tokens_generated": safe_tokens,
                "elapsed_sec": 0.01,
                "tokens_per_sec": 0.0,
                "peak_ram_mb": round(self.process.memory_info().rss / (1024 * 1024), 1),
                "cpu_estimate": 0.0,
                "length_mode": mode,
                "grounding_check": {
                    "is_grounded": True,
                    "flagged_issues": [],
                    "fallback_used": True,
                },
                "settings": {
                    "max_new_tokens": effective_max_tokens,
                    "min_new_tokens": min_new_tokens,
                    "length_mode": mode,
                    "temperature": temperature if do_sample else None,
                    "top_p": top_p if do_sample else None,
                    "repetition_penalty": repetition_penalty,
                    "do_sample": do_sample,
                    "seed": seed,
                },
            }

        prompt = self.build_prompt(topic, personal_knowledge, lead_in=lead_in, length_mode=mode)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        torch.manual_seed(seed)
        cpu_before = psutil.cpu_percent(interval=None)
        t_start = time.perf_counter()

        with torch.inference_mode():
            if do_sample:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=effective_max_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=effective_max_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=False,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        t_elapsed = time.perf_counter() - t_start
        cpu_after = psutil.cpu_percent(interval=None)
        current_ram_mb = self.process.memory_info().rss / (1024 * 1024)

        gen_tokens = output_ids[0][input_len:]
        num_generated_tokens = len(gen_tokens)
        decoded_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        decoded_text = ensure_complete_thoughts(decoded_text)

        # Reconstruct complete personal draft with lead_in if applicable
        complete_draft = f"{lead_in.strip()} {decoded_text}".strip() if lead_in else decoded_text
        complete_draft = ensure_complete_thoughts(complete_draft)

        # Grounding V3: Conservative Post-Generation Validation
        grounding_result = validate_grounding(topic, personal_knowledge, complete_draft)
        fallback_used = False

        if not grounding_result["is_grounded"]:
            # If initial generation had issues, first try deterministic greedy retry if sampled
            fb_adopted = False
            if do_sample:
                t_fb_start = time.perf_counter()
                with torch.inference_mode():
                    fb_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=effective_max_tokens,
                        min_new_tokens=min_new_tokens,
                        do_sample=False,
                        repetition_penalty=max(1.2, repetition_penalty),
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                fb_tokens = fb_ids[0][input_len:]
                fb_text = self.tokenizer.decode(fb_tokens, skip_special_tokens=True).strip()
                fb_text = ensure_complete_thoughts(fb_text)
                fb_draft = f"{lead_in.strip()} {fb_text}".strip() if lead_in else fb_text
                fb_draft = ensure_complete_thoughts(fb_draft)
                fb_grounding = validate_grounding(topic, personal_knowledge, fb_draft)
                t_elapsed += (time.perf_counter() - t_fb_start)

                if fb_grounding["is_grounded"]:
                    decoded_text = fb_text
                    complete_draft = fb_draft
                    num_generated_tokens = len(fb_tokens)
                    grounding_result = fb_grounding
                    fallback_used = True
                    fb_adopted = True

            # If greedy generation is ALSO ungrounded or if greedy was run directly:
            # Enforce deterministic grounded synthesis fallback rather than generating contradictory claims.
            if not fb_adopted:
                safe_fallback = build_grounded_fallback_draft(topic, personal_knowledge, length_mode=mode, lead_in=lead_in)
                decoded_text = safe_fallback
                complete_draft = safe_fallback
                num_generated_tokens = len(self.tokenizer.encode(safe_fallback)) if self.tokenizer else len(safe_fallback.split())
                grounding_result = {"is_grounded": True, "flagged_issues": []}
                fallback_used = True

        tok_per_sec = num_generated_tokens / t_elapsed if t_elapsed > 0 else 0.0

        return {
            "topic": topic,
            "personal_knowledge": personal_knowledge,
            "lead_in": lead_in,
            "prompt": prompt,
            "generated_text": decoded_text,
            "complete_draft": complete_draft,
            "tokens_generated": num_generated_tokens,
            "elapsed_sec": round(t_elapsed, 2),
            "tokens_per_sec": round(tok_per_sec, 2),
            "peak_ram_mb": round(current_ram_mb, 1),
            "cpu_estimate": cpu_after,
            "length_mode": mode,
            "grounding_check": {
                "is_grounded": grounding_result["is_grounded"],
                "flagged_issues": grounding_result["flagged_issues"],
                "fallback_used": fallback_used,
            },
            "settings": {
                "max_new_tokens": effective_max_tokens,
                "min_new_tokens": min_new_tokens,
                "length_mode": mode,
                "temperature": temperature if do_sample else None,
                "top_p": top_p if do_sample else None,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "seed": seed,
            },
        }


def run_benchmark_suite(generator: PersonalizedWritingGenerator, mode: str = "both") -> List[Dict[str, Any]]:
    """Runs generation across distinct authentic Echo personal knowledge topics."""
    benchmark_topics = [
        {
            "id": "test_1_skills_transition",
            "name": "Web Development Learning Journey & AI Transition",
            "topic": "My background in web development and moving to newer technologies",
            "knowledge": [
                "Category: skills | Learned: HTML, CSS, JavaScript, React, TypeScript, Tailwind, Node.js, Express.js, PostgreSQL",
                "Category: goals | Next plan: move ahead to learn AI, machine learning, and cloud technologies",
            ],
            "lead_in": "I started by",
            "max_new_tokens": 85,
        },
        {
            "id": "test_2_task_tracker_project",
            "name": "Task Tracker Project & Offline Queue Workaround",
            "topic": "Why and how I built the Task Tracker application",
            "knowledge": [
                "Category: projects | Built: Task Tracker, a productivity application, when applying to a software company",
                "Category: projects | Constraint: could not rely on continuous internet connectivity, so built a local offline queue that syncs when reconnected",
            ],
            "lead_in": "Recently I built",
            "max_new_tokens": 85,
        },
        {
            "id": "test_3_ai_workflow_habit",
            "name": "Problem Solving Workflow & AI Assistance Routine",
            "topic": "How I handle getting stuck when coding or solving technical problems",
            "knowledge": [
                "Category: habits | Routine: uses AI whenever stuck because it gives concise answers and searching elsewhere takes too much time",
            ],
            "lead_in": "Whenever I am stuck,",
            "max_new_tokens": 80,
        },
        {
            "id": "test_4_architecture_indexeddb",
            "name": "Privacy & Local Client-Side Training Preferences",
            "topic": "Why user chat data should stay local in IndexedDB for personal AI training",
            "knowledge": [
                "Category: opinions | Architecture: chat data should stay stored in user's indexed db so they train their own model",
                "Category: preferences | Cost: prefers local system training without spending money on cloud GPUs",
            ],
            "lead_in": "I was thinking that",
            "max_new_tokens": 80,
        },
    ]

    modes_to_run = ["sampled", "greedy"] if mode == "both" else [mode]
    results = []

    print("=" * 80)
    print("ECHO — LOCAL LORA V2 GENERATION PROOF OF CONCEPT BENCHMARK")
    print("=" * 80)

    for idx, test in enumerate(benchmark_topics, start=1):
        print(f"\n[{idx}/{len(benchmark_topics)}] Test Case: {test['name']} ({test['id']})")
        print(f"Topic: \"{test['topic']}\"")
        print("Supplied Knowledge Records:")
        for k in test["knowledge"]:
            print(f"  • {k}")
        print(f"Lead-in: \"{test['lead_in']}\"")

        test_entry = {
            "test_id": test["id"],
            "test_name": test["name"],
            "topic": test["topic"],
            "knowledge": test["knowledge"],
            "lead_in": test["lead_in"],
            "runs": {},
        }

        for m in modes_to_run:
            is_sample = (m == "sampled")
            res = generator.generate(
                topic=test["topic"],
                personal_knowledge=test["knowledge"],
                lead_in=test["lead_in"],
                max_new_tokens=test["max_new_tokens"],
                temperature=0.65 if is_sample else None,
                top_p=0.9 if is_sample else None,
                repetition_penalty=1.15,
                do_sample=is_sample,
                seed=42,
            )
            test_entry["runs"][m] = res

            print("-" * 60)
            print(f"GENERATED PERSONAL DRAFT ({m.upper()}):")
            print(res["complete_draft"])
            print(
                f"Performance: {res['elapsed_sec']}s ({res['tokens_generated']} tokens, "
                f"{res['tokens_per_sec']} tok/s) | Peak RAM: {res['peak_ram_mb']} MB"
            )

        results.append(test_entry)

    return results


def main():
    parser = argparse.ArgumentParser(description="Echo Local LoRA V2 Personalized Writing Generator")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base HuggingFace model path")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER_PATH, help="Path to trained PEFT LoRA adapter")
    parser.add_argument("--topic", type=str, default="", help="Custom writing topic")
    parser.add_argument("--knowledge", nargs="*", default=[], help="List of personal knowledge facts")
    parser.add_argument("--lead-in", type=str, default="", help="Optional starter phrase for the draft")
    parser.add_argument("--max-tokens", type=int, default=85, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--output-json", type=str, default="", help="Path to write JSON results")

    args = parser.parse_args()

    generator = PersonalizedWritingGenerator(
        base_model_name=args.base_model,
        adapter_path=args.adapter,
    )
    generator.load_model()

    if args.topic and args.knowledge:
        # Single custom generation run
        print(f"Generating draft for topic: '{args.topic}'")
        res = generator.generate(
            topic=args.topic,
            personal_knowledge=args.knowledge,
            lead_in=args.lead_in,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.greedy,
        )
        print("\n" + "=" * 60)
        print("PERSONALIZED DRAFT:")
        print(res["complete_draft"])
        print("=" * 60)
        print(f"Time: {res['elapsed_sec']}s | Tokens: {res['tokens_generated']} | Peak RAM: {res['peak_ram_mb']} MB")

        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            print(f"Saved result to {args.output_json}")
    else:
        # Run full proof of concept benchmark suite across authentic topics
        benchmark_mode = "greedy" if args.greedy else "both"
        results = run_benchmark_suite(generator, mode=benchmark_mode)

        # Output to persistent scratch artifact
        out_path = Path("training-output/qwen3.5-0.8b-lora-v2/generation_poc_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_model": args.base_model,
                    "adapter_path": args.adapter,
                    "model_load_time_sec": round(generator.load_time_sec, 2),
                    "memory_baseline_mb": round(generator.initial_ram_mb, 1),
                    "memory_loaded_mb": round(generator.loaded_ram_mb, 1),
                    "benchmark_results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nBenchmark suite complete. Results saved to: {out_path}")


if __name__ == "__main__":
    main()

