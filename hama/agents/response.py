"""Agent C: response generation via a locally hosted SLM (Ollama).

Answers R1-Q9 / R2 / R3-4 with measurable facts:
  * model: llama3.2:1b (1.24B params, Q8 quantized, ~1.3GB on disk),
    served by Ollama on THIS machine (fog-tier CPU, no cloud call).
  * memory: reported from `ollama ps` (resident model size).
  * latency: wall-clock per response, reported separately from detection
    latency (response generation is asynchronous and event-driven — it runs
    only on alerts, so it does not sit on the detection path).

Severity-adaptive prompt templates follow the manuscript appendix.
"""

import json
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"

PROMPT_LOW = (
    "You are a predictive-maintenance assistant for industrial equipment.\n"
    "Equipment shows a subtle sensor anomaly (severity {severity:.2f} on a 0-1 "
    "scale).\nTop contributing sensors (from SHAP attribution): {top_features}.\n"
    "Dataset/asset context: {context}.\n"
    "In at most 80 words, state: (1) the most likely cause; "
    "(2) whether to continue monitoring or schedule an inspection, with a "
    "concrete timeframe."
)

PROMPT_HIGH = (
    "You are a predictive-maintenance assistant for industrial equipment.\n"
    "CRITICAL anomaly detected (severity {severity:.2f} on a 0-1 scale).\n"
    "Top contributing sensors (from SHAP attribution): {top_features}.\n"
    "Dataset/asset context: {context}.\n"
    "In at most 80 words, state: (1) likely failure mechanism; (2) immediate "
    "recommended action; (3) required technician skills; (4) expected "
    "intervention timeline in hours."
)


def _post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_memory_mb() -> float | None:
    """Resident size of the loaded model, from Ollama's process list."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=5) as r:
            info = json.loads(r.read())
        for m in info.get("models", []):
            if m["name"].startswith(MODEL.split(":")[0]):
                return m["size"] / 1e6
    except Exception:
        return None
    return None


class AgentC:
    def __init__(self, model: str = MODEL, high_cut: float = 0.8):
        self.model = model
        self.high_cut = high_cut
        self.log = []

    def generate(self, severity: float, top_features: list[str], context: str) -> dict:
        template = PROMPT_HIGH if severity >= self.high_cut else PROMPT_LOW
        prompt = template.format(
            severity=severity,
            top_features=", ".join(top_features),
            context=context,
        )
        t0 = time.perf_counter()
        out = _post("/api/generate", {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 160},
        })
        latency_s = time.perf_counter() - t0
        rec = {
            "severity": severity,
            "prompt_template": "high" if severity >= self.high_cut else "low",
            "response": out.get("response", "").strip(),
            "latency_s": latency_s,
            "eval_tokens": out.get("eval_count"),
            "model": self.model,
        }
        self.log.append(rec)
        return rec
