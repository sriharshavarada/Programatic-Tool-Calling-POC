"""CLI wrapper: LangGraph local execution demo."""

from __future__ import annotations

import json

from demo_engine import run_local_langgraph


def main() -> None:
    result = run_local_langgraph()
    for item in result.logs:
        print(item.message)
    print("[LOCAL] ===== SUMMARY =====")
    print(f"[LOCAL] ok={result.ok} duration_ms={result.duration_ms}")
    print(f"[LOCAL] token_usage={json.dumps(result.token_usage, default=str)}")
    print(f"[LOCAL] metadata={json.dumps(result.metadata, default=str)}")
    if result.generated_code:
        print("[LOCAL] ----- GENERATED CODE START -----")
        print(result.generated_code)
        print("[LOCAL] ----- GENERATED CODE END -----")
    if result.final_text:
        print("[LOCAL] Final text:")
        print(result.final_text)
    if result.error:
        print(f"[LOCAL] ERROR: {result.error}")


if __name__ == "__main__":
    main()
