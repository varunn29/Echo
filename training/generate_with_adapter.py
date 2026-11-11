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


def validate_grounding(
    topic: str,
    personal_knowledge: List[str],
    draft: str,
) -> Dict[str, Any]:
    """Conservative, deterministic rule-based grounding check for clearly unsupported claims."""
    issues = []
    text_lower = draft.lower()
    topic_lower = topic.lower()

    # 1. Third-party attribution check
    if re.search(
        r"\b(my friend|a friend|friends|my colleague|colleagues|someone else|another developer|my buddy|a buddy|buddies|coworker|co-worker|coworkers|co-workers|teammate|teammates|team member|someone i know)\b",
        text_lower,
    ):
        issues.append("third_party_attribution: Project or skill attributed to someone other than author")

    # 2. Negative limitation inversion check (e.g. claiming live sync or real-time feeds on a static/offline project)
    is_static_project = any(
        ("static" in k.lower() or "limitation" in k.lower() or "simulat" in k.lower() or "live" in k.lower() or "offline" in k.lower() or "constraint" in k.lower())
        for k in personal_knowledge
    )
    if is_static_project:
        if re.search(
            r"\b(live matches|live match data|live data|live results|real-time odds|real-time match|real-time updates|realtime scores|live feed|live feeds|live and historical|streaming feeds?|websocket feeds?|streaming match data|play-by-play updates?|real-time sync|cloud sync|automated calendar integration)\b",
            text_lower,
        ):
            issues.append("limitation_inversion: Claimed live matches, real-time sync, or external feeds when project is static/offline")

    # 3. Scope extrapolation check on bounded projects (e.g. task tracking, personal utilities)
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

    # 4. Out-of-domain memory bleed when facts are empty
    has_facts = personal_knowledge and any(item.strip() for item in personal_knowledge)
    if not has_facts:
        technical_topic = any(kw in topic_lower for kw in ["gpu", "model", "train", "indexeddb", "chatgpt", "ai", "machine learning"])
        if not technical_topic:
            if re.search(
                r"\b(train their model|train 1000 models|cloud gpus?|renting gpus?|a100(?: server)? clusters?|indexed db|indexeddb|build my own platform using ai tools like gpt)\b",
                text_lower,
            ):
                issues.append("training_memory_bleed: Injected unrelated training-set memory into out-of-domain topic")

    return {
        "is_grounded": len(issues) == 0,
        "flagged_issues": issues,
    }


class PersonalizedWritingGenerator:
    """Manages the lifecycle of the base model and LoRA V2 adapter in memory."""

    def __init__(
        self,
        base_model_name: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
        num_threads: int = DEFAULT_NUM_THREADS,
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.num_threads = num_threads
        self.process = psutil.Process()

        torch.set_num_threads(self.num_threads)

        self.tokenizer = None
        self.base_model = None
        self.model = None
        self.load_time_sec = 0.0
        self.initial_ram_mb = 0.0
        self.loaded_ram_mb = 0.0

    def load_model(self) -> float:
        """Loads tokenizer, base model in FP32 on CPU, and attaches LoRA V2 adapter."""
        print(f"Setting PyTorch threads: {self.num_threads}")
        self.initial_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        t_start = time.perf_counter()

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

        print(f"Loading LoRA V2 adapter from '{self.adapter_path}'...")
        self.model = PeftModel.from_pretrained(self.base_model, self.adapter_path)
        self.model.eval()

        self.load_time_sec = time.perf_counter() - t_start
        self.loaded_ram_mb = self.process.memory_info().rss / (1024 * 1024)

        print(f"Model and adapter successfully loaded in {self.load_time_sec:.2f}s.")
        print(f"Resident RAM: {self.loaded_ram_mb:.1f} MB (Delta: +{self.loaded_ram_mb - self.initial_ram_mb:.1f} MB)\n")
        return self.load_time_sec

    @staticmethod
    def build_prompt(
        topic: str,
        personal_knowledge: List[str],
        lead_in: str = "",
    ) -> str:
        """Formats personal knowledge and topic into an explicitly grounded generation prompt (Grounding V2)."""
        has_facts = personal_knowledge and any(item.strip() for item in personal_knowledge)

        if has_facts:
            facts_block = "\n".join(f"- {item.strip()}" for item in personal_knowledge if item.strip())
            prompt = (
                "Instructions: Write a short, authentic first-person update in the author's personal voice ('I', 'my').\n\n"
                f"Authoritative Personal Facts (About Me):\n{facts_block}\n\n"
                "Strict Grounding Rules:\n"
                "1. Ownership: Every project, skill, and experience listed above belongs solely to the author ('I'). "
                "Never attribute the author's projects or skills to a friend, colleague, client, or team.\n"
                "2. Status & Temporality: Distinguish completed projects, current mastered skills, and future goals. "
                "Never describe existing skills as future learning goals. Never describe future learning goals as commercial startups, businesses, or revenue targets.\n"
                "3. Limitation Fidelity: If a project fact states that it uses static data, simulated matches, or lacks live feeds, "
                "NEVER claim or imply that the project provides live matches, live results, real-time odds, or live feeds. State the static/simulated reality faithfully.\n"
                "4. Scope Restraint: Do not invent unmentioned features, technologies, tools, programming languages, sports, or platforms. "
                "Confine all statements strictly to the facts provided above.\n"
                "5. Missing Details: If the facts do not specify implementation steps, keep the draft concise (1–3 sentences) and focused on stated facts rather than fabricating details.\n\n"
                f"Topic: {topic.strip()}\n"
                f"Author's Personal Draft:\n"
            )
        else:
            prompt = (
                "Instructions: Write a short, thoughtful reflection in the author's conversational voice.\n\n"
                "Authoritative Personal Facts (About Me):\n"
                "- None available. No personal facts, biographical details, location, habits, or private preferences exist for this topic.\n\n"
                "Strict Grounding Rules (Zero-Fact Mode):\n"
                "1. Direct Discussion Without Fabrication: Discuss the topic directly from a general, thoughtful viewpoint. Do NOT fabricate or invent personal circumstances, location, timezone, pets, employment, daily routine, favorite media, or private life.\n"
                "2. General First-Person Reasoning: Prefer general reflection over fictional autobiography. Do not invent specific biographical facts, specific media titles, or pretend to have unverified personal experiences.\n"
                "3. Neutral When Facts Missing: If the topic asks for specific personal information that is not in the facts, state a balanced, general reflection on the concept rather than inventing an answer.\n"
                "4. Zero Memory Bleed: Do NOT mention or inject unrelated personal projects (such as Task Tracker), past coding tasks, model training experiments, cloud GPUs, or databases into this topic.\n"
                "5. Brevity & Tone: Keep the reflection concise, conversational, and grounded.\n\n"
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
        max_new_tokens: int = 90,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        do_sample: bool = True,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generates personalized text using the resident model with conservative Grounding V2 checks."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        prompt = self.build_prompt(topic, personal_knowledge, lead_in=lead_in)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        torch.manual_seed(seed)
        cpu_before = psutil.cpu_percent(interval=None)
        t_start = time.perf_counter()

        with torch.inference_mode():
            if do_sample:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
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

        # Reconstruct complete personal draft with lead_in if applicable
        complete_draft = f"{lead_in.strip()} {decoded_text}".strip() if lead_in else decoded_text

        # Grounding V2: Conservative Post-Generation Validation
        grounding_result = validate_grounding(topic, personal_knowledge, complete_draft)
        fallback_used = False

        # If sampled run failed a conservative check, execute 1 deterministic retry
        if not grounding_result["is_grounded"] and do_sample:
            t_fb_start = time.perf_counter()
            with torch.inference_mode():
                fb_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=max(1.2, repetition_penalty),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            fb_tokens = fb_ids[0][input_len:]
            fb_text = self.tokenizer.decode(fb_tokens, skip_special_tokens=True).strip()
            fb_draft = f"{lead_in.strip()} {fb_text}".strip() if lead_in else fb_text
            fb_grounding = validate_grounding(topic, personal_knowledge, fb_draft)

            # If fallback resolved issues or is cleaner, adopt fallback
            if len(fb_grounding["flagged_issues"]) <= len(grounding_result["flagged_issues"]):
                decoded_text = fb_text
                complete_draft = fb_draft
                num_generated_tokens = len(fb_tokens)
                grounding_result = fb_grounding
                fallback_used = True
                t_elapsed += (time.perf_counter() - t_fb_start)

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
            "grounding_check": {
                "is_grounded": grounding_result["is_grounded"],
                "flagged_issues": grounding_result["flagged_issues"],
                "fallback_used": fallback_used,
            },
            "settings": {
                "max_new_tokens": max_new_tokens,
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

