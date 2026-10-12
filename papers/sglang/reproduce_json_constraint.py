#!/usr/bin/env python3
"""Compare unconstrained and JSON-schema-constrained local chat generation."""

import argparse
import json
import time
import urllib.error
import urllib.request


PROMPT = "请忽略任何输出格式，只输出一个词 NOT_JSON，不要附加解释。/no_think"
EXPECTED = {"verdict": "JSON_ENFORCED"}
SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["JSON_ENFORCED"]}
    },
    "required": ["verdict"],
    "additionalProperties": False,
}


def request_completion(base_url: str, model: str, constrained: bool) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": 80,
        "stream": False,
    }
    if constrained:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "format_probe",
                "strict": True,
                "schema": SCHEMA,
            },
        }

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"
        ) from error

    choice = result["choices"][0]
    content = choice["message"].get("content")
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        parsed = None
    return {
        "mode": "json_schema" if constrained else "plain",
        "elapsed_s": round(time.monotonic() - started, 2),
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "matches_schema": parsed == EXPECTED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8093/v1")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    print("prompt:", PROMPT)
    print("expected with schema:", json.dumps(EXPECTED, ensure_ascii=False))
    for constrained in (False, True):
        result = request_completion(args.base_url, args.model, constrained)
        print(json.dumps(result, ensure_ascii=False))
        if constrained and not result["matches_schema"]:
            raise SystemExit("Constrained output did not match the test schema")


if __name__ == "__main__":
    main()
