import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.scripts import llm_cost as lc


def text_request(chars: int, system_chars: int = 0) -> dict:
    request = {"contents": [{"parts": [{"text": "x" * chars}], "role": "user"}]}
    if system_chars:
        request["systemInstruction"] = {"parts": [{"text": "s" * system_chars}]}
    return request


def image_request(n_images: int, chars: int = 0) -> dict:
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": "AAAA"}}
             for _ in range(n_images)]
    if chars:
        parts.append({"text": "x" * chars})
    return {"contents": [{"parts": parts, "role": "user"}]}


def write_requests(path: Path, requests):
    with open(path, "w") as handle:
        for i, request in enumerate(requests):
            handle.write(json.dumps({"key": f"k{i}", "request": request}) + "\n")


class EstimateTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_counts_text_and_system_instruction(self):
        # The system prompt is billed too, so it must be counted.
        prompt, chars, images = lc.estimate_request(text_request(700, 350))
        self.assertEqual(chars, 1050)
        self.assertEqual(images, 0)
        self.assertEqual(prompt, int(1050 / lc.CHARS_PER_TOKEN))

    def test_counts_images_at_the_ceiling_rate(self):
        prompt, chars, images = lc.estimate_request(image_request(4))
        self.assertEqual(images, 4)
        self.assertEqual(chars, 0)
        self.assertEqual(prompt, 4 * lc.TOKENS_PER_IMAGE)

    def test_accepts_camel_case_inline_data(self):
        request = {"contents": [{"parts": [{"inlineData": {"data": "AA"}}]}]}
        _, _, images = lc.estimate_request(request)
        self.assertEqual(images, 1)

    def test_batch_is_half_of_on_demand(self):
        path = self.root / "r.jsonl"
        write_requests(path, [text_request(4000)] * 10)
        estimate = lc.estimate_jsonl(path)
        self.assertEqual(estimate.n_requests, 10)
        self.assertAlmostEqual(estimate.usd_batch,
                               estimate.usd_on_demand * 0.5, places=6)
        self.assertLess(estimate.usd(online=False),
                        estimate.usd(online=True))

    def test_guarded_cost_exceeds_the_raw_estimate(self):
        path = self.root / "r.jsonl"
        write_requests(path, [text_request(4000)])
        estimate = lc.estimate_jsonl(path)
        self.assertGreater(estimate.guarded_usd(online=True),
                           estimate.usd(online=True))
        self.assertAlmostEqual(
            estimate.guarded_usd(online=True),
            estimate.usd(online=True) * lc.SAFETY_FACTOR, places=6)

    def test_large_prompts_bill_at_the_higher_band(self):
        path = self.root / "r.jsonl"
        # Over 200k prompt tokens: 200k * 3.5 chars.
        write_requests(path, [text_request(800_000)])
        estimate = lc.estimate_jsonl(path)
        self.assertEqual(estimate.n_large_prompts, 1)
        self.assertIn("higher band", estimate.describe(online=True))

    def test_missing_file_estimates_nothing_and_says_so(self):
        estimate = lc.estimate_jsonl(self.root / "absent.jsonl")
        self.assertEqual(estimate.n_requests, 0)
        self.assertEqual(estimate.usd_on_demand, 0.0)
        self.assertTrue(any("does not exist" in n for n in estimate.notes))

    def test_output_tokens_assumption_is_configurable(self):
        path = self.root / "r.jsonl"
        write_requests(path, [text_request(100)] * 3)
        estimate = lc.estimate_jsonl(path, output_tokens_per_request=1000)
        self.assertEqual(estimate.output_tokens, 3000)


class EnforceLimitTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def estimate_costing(self, usd: float) -> lc.Estimate:
        estimate = lc.Estimate(n_requests=1)
        estimate.usd_on_demand = usd
        estimate.usd_batch = usd * 0.5
        return estimate

    def test_under_the_limit_proceeds(self):
        lc.enforce_limit(self.estimate_costing(1.0), limit_usd=50.0,
                         label="cheap", online=True, interactive=False)

    def test_over_the_limit_raises_without_a_terminal(self):
        # An unattended run must not be able to approve its own spend.
        with self.assertRaises(lc.CostLimitExceeded) as ctx:
            lc.enforce_limit(self.estimate_costing(100.0), limit_usd=50.0,
                             label="expensive", online=True, interactive=False)
        message = str(ctx.exception)
        self.assertIn("--approve_cost", message)
        self.assertIn("--cost_limit", message)

    def test_explicit_approval_proceeds(self):
        lc.enforce_limit(self.estimate_costing(100.0), limit_usd=50.0,
                         label="expensive", online=True, approved=True,
                         interactive=False)

    def test_the_safety_factor_is_what_the_limit_sees(self):
        # $42 estimated x 1.25 = $52.50, which is over a $50 limit even though
        # the raw estimate is under it.
        estimate = self.estimate_costing(42.0)
        self.assertLess(estimate.usd(online=True), 50.0)
        with self.assertRaises(lc.CostLimitExceeded):
            lc.enforce_limit(estimate, limit_usd=50.0, label="borderline",
                             online=True, interactive=False)

    def test_batch_transport_is_cheaper_and_can_pass_where_online_fails(self):
        estimate = self.estimate_costing(80.0)  # batch = $40, x1.25 = $50
        with self.assertRaises(lc.CostLimitExceeded):
            lc.enforce_limit(estimate, limit_usd=50.0, label="online",
                             online=True, interactive=False)
        lc.enforce_limit(estimate, limit_usd=50.0, label="batch",
                         online=False, interactive=False)


class ActualFromResultsTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_sums_prompt_output_and_thinking(self):
        path = self.root / "results.jsonl"
        with open(path, "w") as handle:
            handle.write(json.dumps({"key": "a", "response": {
                "usageMetadata": {"promptTokenCount": 100,
                                  "candidatesTokenCount": 10,
                                  "thoughtsTokenCount": 5}}}) + "\n")
            handle.write(json.dumps({"key": "b", "error": "failed"}) + "\n")
        actual = lc.actual_from_results(path)
        self.assertEqual(actual["calls"], 1)
        self.assertEqual(actual["prompt_tokens"], 100)
        # output and thinking bill identically, so they are summed.
        self.assertEqual(actual["output_tokens"], 15)


if __name__ == "__main__":
    unittest.main()
