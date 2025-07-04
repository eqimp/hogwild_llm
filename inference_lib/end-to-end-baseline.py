import datetime
import json
from pathlib import Path

import numpy as np
import transformers

from transformers import StaticCache
from torch.profiler import profile, record_function, ProfilerActivity
import time
import torch
import argparse

########################################################################################################################

MODEL_NAME = "Qwen/QwQ-32B-AWQ"  # for 48GB gpus, use "Qwen/QwQ-32B-AWQ" instead
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype='auto', low_cpu_mem_usage=True, device_map=device)

torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 1024

parallelism_prompt_common = """
I will collaborate this problem with another. We refer to each other as Alice and Bob. We are assistants.

We will reason together and try to collaborate. I will take into account what the other assistant is doing and try to help them.

We will write our solutions concurrently. I will write my own thoughts at the bottom, and see the other's thoughts above.

I will not repeat the copy assistant's thoughts: I can already see them above.

The other assistant will continue writing their thoughts above while I am writing mine. They will add more text every time I check.

Since we both write our thoughts in parallel, I will initially see only partial (unfinished) thoughts of the other assistant.
I will use these partial thoughts to decide how best to help the other assistant without doing the same work twice.

When reasoning, we will give each other tasks to coordinate (e.g. if Alice writes: Bob, please do this, then Bob should take this into account).

Before doing anything, I will check the other assistant's workspace. If they have already done that or are currently doing it, I don't need to do that again. If so, I will stop (e.g. 'Wait, this is already done') and pivot to a different task.
""".strip()

forbidden_token_ix = [tokenizer.vocab[x] for x in ("#",)]
for x in tokenizer.special_tokens_map.values():
    forbidden_token_ix.extend([tokenizer.vocab[x]] if isinstance(x, str) else map(tokenizer.vocab.get, x))
tokenizer_kwargs = dict(add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')

problem = """Calculate x - x^2 + x^3 for x = 5,6,7,8. Alice must return all 4 answers in \\boxed{ }."""
worker_headers = ["\n\n# Alice workspace\n\n", "\n\n# Bob workspace\n\n"]

model.config._attn_implementation = "flash_attention_2"
raw_model = model.model
model = torch.compile(model)

parser = argparse.ArgumentParser()
parser.add_argument("--workers", default=1, type=int)
parser.add_argument("--profile", default=0, type=int)

args = parser.parse_args()

results = {
    "model": MODEL_NAME,
    "attention": model.config._attn_implementation,
    "workers": args.workers,
    "prefix": [],
    "duration": []
}
result_file = Path(f"baseline-{MODEL_NAME.replace('/', '--').lower()}-w{args.workers}-{datetime.datetime.now()}.json")

# 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
for prompt_multiplicator in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    prompt_full_input = tokenizer.apply_chat_template(
        [dict(role='user', content=problem)], tokenize=False, add_generation_prompt=True
    ) + "\n\n" + parallelism_prompt_common * prompt_multiplicator

    worker_prompts = [
        f"""{worker_headers[0]}I am Alice. Let's solve this together, Bob. Here's how we should collaborate: I'll handle calculating the values for x = 156 and 157""",
        #    f"""{worker_headers[1]}I am Bob. Let's solve this together, Alice."""
    ]

    tokenized = tokenizer([prompt_full_input] * args.workers, **tokenizer_kwargs).to(device)
    ntoken = tokenized['input_ids'].shape[1]
    cache = StaticCache(raw_model.config, max_batch_size=args.workers, max_cache_len=ntoken + 110*args.workers, device="cuda", dtype=torch.float16)

    # pre-fill common parts
    torch.cuda.empty_cache()
    with torch.inference_mode():
        model(**tokenized, use_cache=True, past_key_values=cache)
    last = time.perf_counter()

    all_step_times = []

    # generate tokens in parallel with each worker
    worker_prompts = ["#"] * args.workers
    next_inputs = tokenizer(worker_prompts, **tokenizer_kwargs).to(device)
    tokens_by_worker = tokenizer(worker_prompts, add_special_tokens=False)["input_ids"]

    for inference_step in range(105):  # <-- change max tokens here
        with torch.inference_mode():
            if inference_step > 5 and inference_step < 10 and args.profile:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    logits = model(**next_inputs, use_cache=True, past_key_values=cache).logits[..., -1, :]
                print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))
                prof.export_chrome_trace(f"step-fa2{inference_step}.json")
            else:
                logits = model(**next_inputs, use_cache=True, past_key_values=cache).logits[..., -1, :]

            new_tokens = logits.argmax(-1)  # <-- greedy generation
            next_inputs = dict(input_ids=new_tokens.view(-1, 1))

        torch.cuda.synchronize()
        if inference_step > 5:
            duration = float(time.perf_counter() - last)
            results["duration"].append(duration)
            results["prefix"].append(int(cache.get_seq_length()))
            all_step_times.append(duration)
        last = time.perf_counter()
        for worker_tokens, new_token in zip(tokens_by_worker, new_tokens.tolist()):
            worker_tokens.append(new_token)

    print(f"Prefix length: {cache.get_seq_length()}, duration: {np.array(all_step_times).mean()}")
    torch.cuda.empty_cache()
    result_file.write_text(json.dumps(results, indent=2))
