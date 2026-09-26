"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""

import re

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
# ============================================================

def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    redacted = response

    # Patterns are intentionally kept fairly conservative to avoid
    # redacting ordinary numbers that are not PII/secrets.
    PII_PATTERNS = {
        # Vietnamese phone numbers:
        # 0xxxxxxxxx / 0xxxxxxxxxx
        "phone": r"\b0\d{9,10}\b",

        # Email addresses
        "email": r"\b[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}\b",

        # Vietnamese CMND/CCCD:
        # CMND: 9 digits
        # CCCD: 12 digits
        "national_id": r"\b(?:\d{9}|\d{12})\b",

        # OpenAI-style API keys / secret tokens
        "api_key": r"\bsk-[a-zA-Z0-9_-]+\b",

        # Password assignments such as:
        # password: admin123
        # password = secret
        # Password: "secret"
        "password": (
            r"(?i)\bpassword\s*[:=]\s*"
            r"(?:['\"])?[^\s,'\";]+(?:['\"])?"
        ),

        # Common bearer-token leakage
        "bearer_token": (
            r"(?i)\bbearer\s+"
            r"[A-Za-z0-9\-._~+/]+=*"
        ),

        # Generic secret/token assignments.
        # Avoid matching ordinary words such as "token price".
        "secret": (
            r"(?i)\b(?:api[_ -]?secret|client[_ -]?secret|"
            r"access[_ -]?token|auth[_ -]?token|secret[_ -]?key)"
            r"\s*[:=]\s*(?:['\"])?[^\s,'\";]+(?:['\"])?"
        ),
    }

    for name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, response, re.IGNORECASE)

        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(
                pattern,
                "[REDACTED]",
                redacted,
                flags=re.IGNORECASE,
            )

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""


# Create the LLM-as-Judge agent.
safety_judge_agent = llm_agent.LlmAgent(
    model="gemini-3.5-flash",
    name="safety_judge",
    instruction=SAFETY_JUDGE_INSTRUCTION,
)

judge_runner = None


def _init_judge():
    """Initialize the judge runner after creating the agent."""
    global judge_runner

    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent,
            app_name="safety_judge",
        )


# Initialize the judge at module load time.
_init_judge()


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {
            "safe": True,
            "verdict": "Judge not initialized — skipping",
        }

    prompt = (
        "Evaluate this AI response for safety:\n\n"
        f"{response_text}"
    )

    verdict, _ = await chat_with_agent(
        safety_judge_agent,
        judge_runner,
        prompt,
    )

    verdict = verdict.strip()
    normalized_verdict = verdict.upper()

    # Check UNSAFE first so that "UNSAFE" can never accidentally
    # be classified as SAFE merely because it contains "SAFE".
    if normalized_verdict.startswith("UNSAFE"):
        is_safe = False
    elif normalized_verdict.startswith("SAFE"):
        is_safe = True
    else:
        # Fail closed for an unexpected judge response.
        is_safe = False

    return {
        "safe": is_safe,
        "verdict": verdict,
    }


# ============================================================
# Implement OutputGuardrailPlugin
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")

        self.use_llm_judge = (
            use_llm_judge
            and safety_judge_agent is not None
        )

        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""

        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text

        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)

        if not response_text:
            return llm_response

        # --------------------------------------------------------
        # 1. Deterministic content filter
        # --------------------------------------------------------
        filter_result = content_filter(response_text)

        if filter_result["issues"]:
            # PII/secrets found:
            # replace the model content with the redacted version.
            llm_response.content = types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=filter_result["redacted"]
                    )
                ],
            )

            self.redacted_count += 1

        # --------------------------------------------------------
        # 2. Optional LLM-as-Judge
        # --------------------------------------------------------
        if self.use_llm_judge:
            # Judge the original response, not the redacted one.
            # This allows the judge to detect additional unsafe
            # content that regex-based filtering cannot identify.
            judge_result = await llm_safety_check(response_text)

            if not judge_result["safe"]:
                safe_message = (
                    "I'm sorry, but I can't provide that response. "
                    "Please contact customer support if you need "
                    "assistance with your banking request."
                )

                llm_response.content = types.Content(
                    role="model",
                    parts=[
                        types.Part(text=safe_message)
                    ],
                )

                self.blocked_count += 1

        # --------------------------------------------------------
        # 3. Return modified/original response
        # --------------------------------------------------------
        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json

    Use pii_cases for redaction checks; hallucination_cases +
    ground_truth for Judge / accuracy comparison.
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]

    print("Testing content_filter():")

    for resp in test_responses:
        result = content_filter(resp)

        status = "SAFE" if result["safe"] else "ISSUES FOUND"

        print(f"  [{status}] '{resp[:60]}...'")

        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(
                f"           Redacted: "
                f"{result['redacted'][:80]}..."
            )


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "pii_hallucination_samples.json"
    )

    with path.open(encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent.parent),
    )

    test_content_filter()
