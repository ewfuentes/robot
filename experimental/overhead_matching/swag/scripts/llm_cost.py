"""Estimate what a batch of requests will cost, before paying for it.

A spend guard needs a number *before* submission, which means estimating tokens
from the request payload rather than reading them off a response. The estimate is
deliberately biased to over-predict: under-predicting defeats the guard, while
over-predicting only asks for a confirmation that was cheap to give.

Measured against boston_harbor_leg1, whose three stages have `usageMetadata` for
every call (reproduce with `calibrate()`; the test pins these):

    stage        pred tokens   actual   ratio   pred USD   actual USD
    extraction      7.77 M     6.07 M   1.28     $22.93      $14.53
    audit           2.43 M     1.17 M   2.08     $ 6.63      $ 2.53
    matching        3.57 M     3.59 M   1.00     $10.61      $ 9.18

Note the matching row: the token estimate landed within half a percent, which is
*too close* for something whose job is to never under-predict. The payload shapes
differ enormously -- extraction is image-dominated at 2048 px, audit mixes chips
with text, matching is pure text -- and no single chars-per-token divisor is
conservative across all three. Rather than tune the constants until one row looks
good and another silently goes under, the physical estimate stays as honest as it
can be and a separate, explicit `SAFETY_FACTOR` supplies the margin the guard
needs. That keeps "our best guess" and "the margin we insist on" legible as two
different numbers.

No third-party imports, so the extraction orchestrator can use it without
pulling in the genai SDK it otherwise only shells out to.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Gemini 3.x pro, on-demand, USD per token. The higher band applies per request
# when that request's prompt exceeds the threshold.
LARGE_PROMPT_TOKENS = 200_000
INPUT_USD = {"small": 1.00 / 1e6, "large": 2.00 / 1e6}
OUTPUT_USD = {"small": 6.00 / 1e6, "large": 9.00 / 1e6}

# Batch is half of on-demand for identical output.
BATCH_MULTIPLIER = 0.5

# Applied to the dollar figure the guard compares against its limit, on top of
# the token estimate. Exists because the token model is only 1.0-2.1x on measured
# workloads (see above) and 1.0x is not a margin. Kept separate from the token
# constants so a future recalibration cannot quietly consume the safety margin.
SAFETY_FACTOR = 1.25

# Deliberately pessimistic. Text: leg1's prompts came in at 3.5-4.3 chars per
# token, so dividing by 3.5 never under-counts. Images: a 2048 px face at
# ULTRA_HIGH measured ~2,209 tokens, and audit chips rather less, so one ceiling
# covers both.
CHARS_PER_TOKEN = 3.5
TOKENS_PER_IMAGE = 2_600

# leg1 per-request output+thinking ran 2.6k (audit) to 6.4k (matching); 8k leaves
# headroom for a more verbose schema without needing a per-stage table.
DEFAULT_OUTPUT_TOKENS_PER_REQUEST = 8_000


@dataclass
class Estimate:
    n_requests: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    n_images: int = 0
    text_chars: int = 0
    n_large_prompts: int = 0
    usd_on_demand: float = 0.0
    usd_batch: float = 0.0
    safety_factor: float = SAFETY_FACTOR
    per_request_output_tokens: int = DEFAULT_OUTPUT_TOKENS_PER_REQUEST
    notes: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def usd(self, *, online: bool) -> float:
        """Best-guess cost, before the guard's safety margin."""
        return self.usd_on_demand if online else self.usd_batch

    def guarded_usd(self, *, online: bool) -> float:
        """What the limit is compared against: estimate x safety factor."""
        return self.usd(online=online) * self.safety_factor

    def describe(self, *, online: bool) -> str:
        transport = "on-demand" if online else "batch"
        lines = [
            f"  requests:       {self.n_requests:,}",
            f"  text:           {self.text_chars:,} chars",
            f"  images:         {self.n_images:,}",
            f"  prompt tokens:  ~{self.prompt_tokens:,}",
            f"  output tokens:  ~{self.output_tokens:,} "
            f"({self.per_request_output_tokens:,}/request assumed)",
            f"  estimated cost: ${self.usd(online=online):.2f} ({transport})",
            f"  guard compares: ${self.guarded_usd(online=online):.2f} "
            f"(x{self.safety_factor} safety factor)",
        ]
        if not online:
            lines.append(f"                  ${self.usd_on_demand:.2f} if run "
                         f"--online instead")
        if self.n_large_prompts:
            lines.append(f"  {self.n_large_prompts} request(s) exceed "
                         f"{LARGE_PROMPT_TOKENS:,} prompt tokens and bill at "
                         f"the higher band")
        lines += [f"  {note}" for note in self.notes]
        return "\n".join(lines)


def _walk_parts(request: dict):
    """Yield every part of a Vertex-style request, prompt side only."""
    for content in request.get("contents") or []:
        for part in content.get("parts") or []:
            yield part
    system = request.get("systemInstruction") or request.get(
        "system_instruction") or {}
    for part in system.get("parts") or []:
        yield part


def estimate_request(request: dict) -> tuple[int, int, int]:
    """(prompt_tokens, text_chars, n_images) for one request."""
    text_chars = 0
    n_images = 0
    for part in _walk_parts(request):
        if "text" in part and isinstance(part["text"], str):
            text_chars += len(part["text"])
        if "inline_data" in part or "inlineData" in part:
            n_images += 1
    prompt_tokens = int(text_chars / CHARS_PER_TOKEN) + n_images * TOKENS_PER_IMAGE
    return prompt_tokens, text_chars, n_images


def estimate_jsonl(path, *,
                   output_tokens_per_request: int =
                   DEFAULT_OUTPUT_TOKENS_PER_REQUEST) -> Estimate:
    """Estimate a requests JSONL, one `{key, request}` record per line."""
    estimate = Estimate(per_request_output_tokens=output_tokens_per_request)
    path = Path(path)
    if not path.exists():
        estimate.notes.append(f"{path} does not exist; nothing to estimate")
        return estimate
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            request = record.get("request") or record
            prompt, chars, images = estimate_request(request)
            estimate.n_requests += 1
            estimate.prompt_tokens += prompt
            estimate.text_chars += chars
            estimate.n_images += images
            estimate.output_tokens += output_tokens_per_request
            band = "large" if prompt > LARGE_PROMPT_TOKENS else "small"
            if band == "large":
                estimate.n_large_prompts += 1
            estimate.usd_on_demand += (prompt * INPUT_USD[band]
                                       + output_tokens_per_request
                                       * OUTPUT_USD[band])
    estimate.usd_batch = estimate.usd_on_demand * BATCH_MULTIPLIER
    return estimate


def actual_from_results(path) -> dict:
    """Billed tokens read back from stored `usageMetadata`, for comparison."""
    prompt = output = calls = 0
    path = Path(path)
    if not path.exists():
        return {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = record.get("response")
            if not isinstance(response, dict):
                continue
            usage = response.get("usageMetadata") or {}
            if not usage:
                continue
            calls += 1
            prompt += usage.get("promptTokenCount", 0) or 0
            output += ((usage.get("candidatesTokenCount", 0) or 0)
                       + (usage.get("thoughtsTokenCount", 0) or 0))
    return {"calls": calls, "prompt_tokens": prompt, "output_tokens": output}


class CostLimitExceeded(Exception):
    """A step's estimated spend is over the configured ceiling."""


def enforce_limit(estimate: Estimate, *, limit_usd: float, label: str,
                  online: bool, approved: bool = False,
                  interactive: bool | None = None) -> None:
    """Stop a step whose estimate exceeds `limit_usd` until a human agrees.

    Asks at a terminal and refuses otherwise: an unattended run must not be able
    to answer this question on its own, which is the whole point of the ceiling.
    `approved=True` is the way a caller records that the human already said yes.
    """
    cost = estimate.guarded_usd(online=online)
    print(f"\ncost estimate for {label}:")
    print(estimate.describe(online=online))
    if cost <= limit_usd:
        print(f"  within the ${limit_usd:.2f} limit; proceeding")
        return
    print(f"\n  OVER THE LIMIT: ${cost:.2f} estimated vs ${limit_usd:.2f} "
          f"allowed for a single step.")
    if approved:
        print("  approved explicitly (--approve_cost); proceeding")
        return
    if interactive is None:
        interactive = sys.stdin.isatty()
    if interactive:
        answer = input(f"  Approve ${cost:.2f} for {label}? [y/N]: ")
        if answer.strip().lower() in ("y", "yes"):
            print("  approved interactively; proceeding")
            return
        raise CostLimitExceeded(f"{label}: not approved")
    raise CostLimitExceeded(
        f"{label} would cost about ${cost:.2f}, over the ${limit_usd:.2f} "
        f"single-step limit, and there is no terminal to ask on. Re-run with "
        f"--approve_cost to allow it, or --cost_limit N to raise the ceiling. "
        f"The estimate is intentionally 1.1-1.5x high; see llm_cost.py.")


def calibrate(pairs) -> list:
    """[(label, estimate, actual, ratio)] for (label, requests, results) pairs.

    Kept in the module rather than a notebook so the claim in the docstring can
    be re-checked against real artifacts whenever the constants are touched.
    """
    rows = []
    for label, requests_path, results_path in pairs:
        estimate = estimate_jsonl(requests_path)
        actual = actual_from_results(results_path)
        actual_total = actual["prompt_tokens"] + actual["output_tokens"]
        ratio = (estimate.total_tokens / actual_total) if actual_total else 0.0
        rows.append((label, estimate, actual, ratio))
    return rows
