"""Web UI for comparing three execution styles of AI task execution."""

from __future__ import annotations

import json
import os

from flask import Flask, jsonify, render_template, request

from demo_engine import run_mode


app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/run/<mode>")
def run(mode: str):
    body = request.get_json(silent=True) or {}
    model = body.get("model") or os.environ.get("ANTHROPIC_MODEL")
    result = run_mode(mode, model=model)
    response = result.to_dict()
    status = 200 if result.ok else 500
    return app.response_class(
        response=json.dumps(response),
        status=status,
        mimetype="application/json",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=False)
