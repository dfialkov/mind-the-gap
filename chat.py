import torch
import gradio as gr
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen3-14B"
DEVICE = "mps"

print(f"Loading {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
).to(DEVICE)
model.eval()
print("Loaded.")


def chat(user_message, history):
    messages = history + [{"role": "user", "content": user_message}]

    print(f"raw messages:\n {messages}")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    print(f"raw prompt: \n{prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        ),
    ).start()

    partial = ""
    for chunk in streamer:
        partial += chunk
        yield partial


gr.ChatInterface(chat, title="Qwen3-14B").launch()
