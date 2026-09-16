"""One definition of a reportable evolution round, shared by runner and reports."""
def invalid_reason(result):
    if result.get("crash"):
        return "generation_crash"
    if any(r.get("tool") == "reload_engine_methods" and not r.get("ok")
           for r in result.get("tool_results", [])):
        return "compile_failed"
    task = result.get("task")
    if task is None:
        return "no_fresh_evaluation"
    if task.get("engine_load_error"):
        return "compile_failed"
    if not task.get("llm_forward_calls", 0):
        return "no_llm_forward_calls"
    if not task.get("episodes") or any(e.get("info", {}).get("error") for e in task["episodes"]):
        return "evaluation_error"
    return None
