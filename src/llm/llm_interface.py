# src/llm/llm_interface.py
# Enhanced with Token Usage tracking

import re
from dataclasses import dataclass

import ollama
import torch


@dataclass
class TokenUsage:
    """Tracks token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMInterface:
    def __init__(self, model_choice="llama3"):
        """
        Initialize Ollama interface with session-level token tracking.
        """
        if model_choice == "auto":
            if torch.cuda.is_available():
                print("⚡ GPU detected. Using Ollama model 'mistral'")
                self.model_name = "mistral"
            else:
                print("💻 CPU detected. Using Ollama model 'llama3'")
                self.model_name = "llama3"
        else:
            self.model_name = model_choice

        # Session-level token accumulators
        self._session_total_tokens = 0
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0
        self._session_llm_calls = 0

        print(f"[INFO] Using Ollama model: {self.model_name}")

    def generate(self, prompt: str) -> str:
        """
        Use Ollama's chat endpoint for generation (backward compatible).
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return f"⚠️ Ollama generation failed: {e}"

    def generate_with_tokens(self, prompt: str) -> tuple[str, TokenUsage]:
        """
        Generate response and capture token usage from Ollama.
        Returns (response_text, TokenUsage)
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response["message"]["content"]

            prompt_tokens = response.get("prompt_eval_count", 0) or 0
            completion_tokens = response.get("eval_count", 0) or 0
            total_tokens = prompt_tokens + completion_tokens

            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            self._session_total_tokens += total_tokens
            self._session_prompt_tokens += prompt_tokens
            self._session_completion_tokens += completion_tokens
            self._session_llm_calls += 1

            return text, usage

        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return f"⚠️ Ollama generation failed: {e}", TokenUsage()

    def generate_defect_title(self, issue_details: str, requirement_id: str) -> tuple[str, TokenUsage]:
        """
        Generate a concise defect title from issue details.

        Title style:
        [Component] Brief description of issue
        """
        cleaned_issue = (issue_details or "").strip()
        cleaned_req = (requirement_id or "").strip()

        if not cleaned_issue:
            return "", TokenUsage()

        prompt = (
            "You are a QA defect triage assistant. "
            "Generate ONE concise defect title using best practices.\n"
            "Rules:\n"
            "1) Keep it clear and specific\n"
            "2) Prefer format: [Component] Brief issue description\n"
            "3) 8 to 14 words ideally\n"
            "4) No markdown, no bullets, no extra explanation\n"
            "5) Output only the title text\n\n"
            f"Requirement ID: {cleaned_req or 'Not provided'}\n"
            f"Issue details:\n{cleaned_issue}\n\n"
            "Defect Title:"
        )

        raw_title, usage = self.generate_with_tokens(prompt)
        title = (raw_title or "").strip().splitlines()[0] if raw_title else ""

        # Cleanup common model artifacts
        title = re.sub(r"^[\-\*\d\.)\s]+", "", title).strip()
        title = title.strip('"').strip("'").strip()

        # Fallback title if model fails or gives empty output
        if not title or title.startswith("⚠️") or "generation failed" in title.lower():
            short_issue = cleaned_issue[:80].strip()
            title = f"[General] {short_issue}" if short_issue else "[General] Defect identified"

        return title, usage

    def get_session_stats(self) -> dict:
        """Return session-level token statistics."""
        return {
            "total_tokens": self._session_total_tokens,
            "prompt_tokens": self._session_prompt_tokens,
            "completion_tokens": self._session_completion_tokens,
            "llm_calls": self._session_llm_calls,
            "avg_tokens_per_query": (
                self._session_total_tokens / self._session_llm_calls
                if self._session_llm_calls > 0 else 0
            ),
        }

    def reset_session_stats(self):
        """Reset all session-level token accumulators."""
        self._session_total_tokens = 0
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0
        self._session_llm_calls = 0
