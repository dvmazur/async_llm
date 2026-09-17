"""Eliza OpenRouter gateway transport; credentials come from the environment."""

import argparse
import json
import os
import ssl
import urllib.error
import urllib.request


ENDPOINT = "https://api.eliza.yandex.net/openrouter/v1/chat/completions"


def request(payload=None, *, timeout=7200, insecure=False, list_models=False):
    ledger = os.environ.get("DATASET_BUDGET_LEDGER")
    if ledger and not list_models:
        from budget import call
        return call(ledger, payload, lambda: _request(payload, timeout=timeout, insecure=insecure))
    return _request(payload, timeout=timeout, insecure=insecure, list_models=list_models)


def _request(payload=None, *, timeout=7200, insecure=False, list_models=False):
    key = os.environ.get("API_KEY")
    if not key:
        raise ValueError("API_KEY is missing; source the repository .env first.")
    url = ENDPOINT.rsplit("/chat/completions", 1)[0] + "/models" if list_models else ENDPOINT
    headers = {"Authorization": f"OAuth {key}", "Ya-Pool": "YR_all",
               "Content-Type": "application/json", "X-Request-Timeout": "60m"}
    req = urllib.request.Request(url, headers=headers,
                                 data=None if list_models else json.dumps(payload).encode())
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
            result = json.load(response)
            # Eliza may wrap the upstream result; never persist the envelope's key field.
            if isinstance(result, dict) and "response" in result:
                inner = result["response"]
                if isinstance(inner, str):
                    try:
                        inner = json.loads(inner)
                    except ValueError:
                        raise RuntimeError("Gateway returned a non-JSON upstream response.") from None
                if not isinstance(inner, dict):
                    raise RuntimeError("Gateway returned an unexpected upstream response type.")
                return inner
            return result
    except urllib.error.HTTPError as exc:
        # Do not print raw gateway responses, which may contain request details.
        raise RuntimeError(f"Gateway HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gateway connection failed: {type(exc.reason).__name__}") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true", required=True)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--filter", default="gemini")
    args = parser.parse_args()
    result = request(list_models=True, timeout=30, insecure=args.insecure)
    if not isinstance(result.get("data"), list):
        print(json.dumps({"catalog_available": False, "response_keys": list(result)}))
        return
    rows = result.get("data", [])
    print(json.dumps({"model_count": len(rows), "models": [
        {"id": row["id"], "input": row.get("architecture", {}).get("input_modalities"),
         "prompt_per_million": float(row.get("pricing", {}).get("prompt", 0)) * 1e6,
         "completion_per_million": float(row.get("pricing", {}).get("completion", 0)) * 1e6}
        for row in rows if args.filter in row.get("id", "").lower()]}, indent=2))


if __name__ == "__main__":
    main()
