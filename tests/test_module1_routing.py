import os
import unittest
from unittest.mock import patch

import feedback_pipeline as fp


class RoutingConfigTests(unittest.TestCase):
    def test_import_does_not_require_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                fp.ensure_api_key()

    def test_default_routing_uses_current_cost_aware_models(self):
        routing = fp.build_model_routing()

        self.assertEqual(routing.generation, "gpt-5.6-terra")
        self.assertEqual(routing.scoring, "gpt-5.6-terra")
        self.assertEqual(routing.verification, "gpt-5.6-terra")
        self.assertEqual(routing.rewrite, "gpt-5.6-luna")
        self.assertEqual(routing.clustering, "gpt-5.6-luna")
        self.assertEqual(routing.meta_review, "gpt-5.6-sol")
        self.assertEqual(routing.escalation, "gpt-5.6-sol")

    def test_generation_override_does_not_change_other_stages(self):
        routing = fp.build_model_routing(gen_model="gpt-5.6-sol")

        self.assertEqual(routing.generation, "gpt-5.6-sol")
        self.assertEqual(routing.scoring, "gpt-5.6-terra")
        self.assertEqual(routing.rewrite, "gpt-5.6-luna")
        self.assertEqual(routing.meta_review, "gpt-5.6-sol")

    def test_invalid_model_is_rejected(self):
        with self.assertRaises(ValueError):
            fp.build_model_routing(gen_model="not-a-model")

    def test_current_model_options_put_current_models_first(self):
        options = fp.current_model_options()

        self.assertLess(options.index("gpt-5.6-sol"), options.index("gpt-5.5"))
        self.assertLess(options.index("gpt-5.6-terra"), options.index("gpt-5.4-mini"))
        self.assertLess(options.index("gpt-5.6-luna"), options.index("gpt-5.4-nano"))

    def test_chat_request_kwargs_preserve_effective_reasoning_defaults(self):
        messages = [{"role": "user", "content": "Review this."}]

        terra = fp._chat_request_kwargs(messages, "gpt-5.6-terra")
        luna = fp._chat_request_kwargs(messages, "gpt-5.6-luna")
        sol = fp._chat_request_kwargs(messages, "gpt-5.6-sol")
        legacy = fp._chat_request_kwargs(messages, "gpt-5.4-mini")

        self.assertEqual(terra["reasoning_effort"], "none")
        self.assertEqual(luna["reasoning_effort"], "none")
        self.assertEqual(sol["reasoning_effort"], "medium")
        self.assertNotIn("reasoning_effort", legacy)


class StructuredOutputTests(unittest.TestCase):
    def test_json_schema_response_format(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        payload = fp.json_schema_response_format("answer_schema", schema)

        self.assertEqual(payload["type"], "json_schema")
        self.assertEqual(payload["json_schema"]["name"], "answer_schema")
        self.assertIs(payload["json_schema"]["schema"], schema)
        self.assertTrue(payload["json_schema"]["strict"])


class CostEstimateTests(unittest.TestCase):
    def test_estimate_uses_routing_models_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            estimate = fp.estimate_cost_before_run(
                "Title\n\nThis paper estimates a difference-in-differences model.",
                num_agents=8,
                top_k=3,
            )

        stages = estimate["stages"]
        self.assertEqual(stages["generation"]["model"], "gpt-5.6-terra")
        self.assertEqual(stages["scoring"]["model"], "gpt-5.6-terra")
        self.assertEqual(stages["score_escalation"]["model"], "gpt-5.6-sol")
        self.assertEqual(stages["verification"]["model"], "gpt-5.6-terra")
        self.assertEqual(stages["rewrite"]["model"], "gpt-5.6-luna")
        self.assertNotIn("critique", stages)
        self.assertNotIn("re_scoring", stages)
        self.assertEqual(stages["clustering"]["model"], "gpt-5.6-luna")
        self.assertEqual(stages["meta_review"]["model"], "gpt-5.6-sol")
        self.assertGreater(estimate["estimated_total_cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
