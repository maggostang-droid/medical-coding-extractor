"""Erzeugt data/demo_examples.json: die 81 Testnotizen mit den ECHTEN
Vorhersagen beider Ansätze aus dem Auswertungslauf.

Damit lädt die Demo Beispiele sofort — ohne Modell im Speicher. Das ist keine
Attrappe: es sind exakt die Ausgaben, aus denen auch die Metriken in
results/results.md berechnet wurden.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

test = read_jsonl(ROOT / "data/test.jsonl")
rag = {r["text"]: r["predicted_codes"] for r in read_jsonl(ROOT / "results/predictions_rag.jsonl")}
ft = {r["text"]: r["predicted_codes"] for r in read_jsonl(ROOT / "results/predictions_finetune.jsonl")}

out = []
for row in test:
    t = row["text"]
    if t not in rag or t not in ft:
        continue
    out.append({
        "text": t,
        "expected": row["expected_codes"],
        "rag": rag[t],
        "finetune": ft[t],
        "difficulty": row.get("difficulty", "?"),
    })

(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data/demo_examples.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(out)} Beispiele geschrieben nach data/demo_examples.json")
