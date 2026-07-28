"""Lädt Llama-3.2-3B-Instruct (optional mit LoRA-Adapter) und generiert
GOZ-Code-Listen. Gemeinsam genutzt vom Colab-Notebook (Training + Baseline-
/Finetune-Inferenz über das Test-Set) und der Streamlit-Demo (Einzelanfragen)."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from goz_extract.prompting import build_extraction_prompt, parse_code_list_response
from goz_extract.schema import GozCode


def load_model(
    model_id: str,
    adapter_path: str | None = None,
    torch_dtype=torch.float16,
    device_map="auto",
):
    """Lädt Modell + Tokenizer.

    Default-dtype ist float16, nicht bfloat16: T4-GPUs (compute capability
    7.5, die Colab-Stufe, auf die dieses Projekt zielt) unterstützen bf16
    nicht nativ - es fällt sonst still auf langsame Emulation zurück, und
    die bitsandbytes-Doku empfiehlt fp16 für Pre-Ampere-Karten. Aufrufer mit
    besserer Hardware (Ampere+) können explizit torch.bfloat16 übergeben.

    device_map="auto" kann bei knappem freiem VRAM Layer auf CPU/Platte
    auslagern (accelerate-Offloading) - das kollidiert live beobachtet mit
    peft's Adapter-Ladelogik (KeyError mit einem zusätzlichen "model."-
    Präfix in _update_offload). Bei adapter_path!=None und wenig freiem
    Speicher: device_map={"": 0} übergeben, um alles auf eine GPU zu
    zwingen (kein Offload) - Llama 3.2 3B passt in fp16 (~6GB) meist auch
    ohne "auto" komplett auf eine T4.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device_map
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
    # tokenize=False + separater tokenizer(...)-Aufruf statt
    # apply_chat_template(..., return_tensors="pt") direkt: letzteres liefert
    # je nach transformers-Version einen reinen Tensor oder ein
    # BatchEncoding-Objekt zurück, was model.generate() unvorhersehbar
    # crashen lässt (fehlender .shape-Zugriff). Der Umweg über
    # tokenizer(text, return_tensors="pt") liefert immer ein BatchEncoding,
    # entpackt via **inputs - robust gegenüber der transformers-Version.
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return parse_code_list_response(generated, valid_codes)
