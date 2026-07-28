"""Lädt Llama-3.2-3B-Instruct (optional mit LoRA-Adapter) und generiert
GOZ-Code-Listen. Gemeinsam genutzt vom Colab-Notebook (Training + Baseline-
/Finetune-Inferenz über das Test-Set) und der Streamlit-Demo (Einzelanfragen)."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from goz_extract.prompting import build_extraction_prompt, parse_code_list_response
from goz_extract.schema import GozCode


def load_model(model_id: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_codes(
    model,
    tokenizer,
    note_text: str,
    valid_codes: set[str],
    candidates: list[GozCode] | None = None,
    max_new_tokens: int = 64,
) -> list[str]:
    prompt = build_extraction_prompt(note_text, candidates=candidates)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tokenizer.decode(output_ids[0][inputs.shape[1]:], skip_special_tokens=True)
    return parse_code_list_response(generated, valid_codes)
