"""Offline MATH grading: only the final complete boxed expression."""


def boxed(text):
    """Last complete boxed expression, including nested LaTeX braces."""
    end = len(text)
    while (start := text.rfind(r"\boxed{", 0, end)) >= 0:
        depth = 1
        for i in range(start + 7, end):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                return text[start + 7:i].strip()
        end = start
    return None


def grade_one(record):
    prediction = boxed(record["writer_text"])
    if prediction is None:
        return dict(correct=False, prediction=None, grading_error=None)
    from math_verify import LatexExtractionConfig, parse, verify

    reference = parse("$" + record["answer"] + "$", extraction_config=[LatexExtractionConfig()],
                      fallback_mode="no_fallback")
    actual = parse("$" + prediction + "$", extraction_config=[LatexExtractionConfig()],
                   fallback_mode="no_fallback")
    if not reference:
        return dict(correct=None, prediction=prediction, grading_error="unparseable_reference")
    return dict(correct=bool(actual) and bool(verify(reference, actual)),
                prediction=prediction, grading_error=None)
