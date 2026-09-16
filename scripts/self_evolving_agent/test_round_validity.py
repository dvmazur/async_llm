import unittest
from round_validity import invalid_reason

class ValidRoundTests(unittest.TestCase):
    def valid(self):
        return {"task": {"llm_forward_calls": 1, "episodes": [{"steps": 0, "reward": 0, "info": {"decision_cancelled": True}}]}, "tool_results": []}
    def test_timeout_before_first_completed_action_is_valid_if_llm_called(self):
        self.assertIsNone(invalid_reason(self.valid()))
    def test_no_forward_not_counted(self):
        result=self.valid(); result["task"]["llm_forward_calls"]=0
        self.assertEqual(invalid_reason(result), "no_llm_forward_calls")
    def test_compile_failure_not_counted_even_with_task_result(self):
        result=self.valid(); result["tool_results"]=[{"tool":"reload_engine_methods", "ok":False}]
        self.assertEqual(invalid_reason(result), "compile_failed")
    def test_no_fresh_evaluation_not_counted(self):
        self.assertEqual(invalid_reason({}), "no_fresh_evaluation")
    def test_errors_not_counted_as_valid_zero_scores(self):
        result=self.valid(); result["task"]["episodes"][0]["info"]={"error":"broken engine"}
        self.assertEqual(invalid_reason(result), "evaluation_error")

if __name__ == '__main__': unittest.main()
