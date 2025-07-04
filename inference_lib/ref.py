from typing import Tuple

import torch
import math
import numpy as np


def hogwild_sdpa(queries: torch.Tensor, locations: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor],
                 scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qk = []
    vals = []
    for f in range(len(keys)):
        # GQA replication
        key = keys[f].repeat_interleave(queries.size(-3) // keys[f].size(-3), -3)
        val = values[f].repeat_interleave(queries.size(-3) // values[f].size(-3), -3)
        # key and query position -> mask
        kp = torch.arange(0, key.size(-2), dtype=torch.int32, device=key.device)
        qp = locations[f]
        kp = kp[None, None, None, :]
        qp = qp[:, None, :, None]
        mask = kp > qp
        att = queries[f].to(torch.float32) @ key.transpose(-1, -2).to(torch.float32)
        if f == 0:
            print(key[0, 0, :])
            print(queries[f, 0, 0, 0, :])
        att.masked_fill_(mask, float("-inf"))
        qk.append(att)
        vals.append(val)

    qk = torch.concat(qk, dim=-1)
    vals = torch.concat(vals, dim=-2).to(torch.float32)
    att = torch.softmax(scale * qk, dim=-1)
    result = att @ vals
    return result, qk, att


@torch.no_grad()
def generate_test_data(file_name: str, F: int, W: int, Hq: int, Hkv: int, E: int, Ev: int, S: int, # noqa
                       frags: list[int], scale: float = None, dtype: str = "fp32"):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 1       # version
    header[1:8] = [F, W, Hq, Hkv, E, Ev, S]
    if dtype == "fp32":
        header[8] = 0
        dtype = torch.float32
    elif dtype == "fp16":
        header[8] = 1
        dtype = torch.float16
    elif dtype == "bf16":
        header[8] = 2
        dtype = torch.bfloat16

    torch.random.manual_seed(42)
    # TODO make input distributions more interesting
    if scale is None:
        scale = 1.0 / math.sqrt(E)
    queries = torch.rand((F, W, Hq, S, E)).to(dtype).to(dtype)
    keys = [torch.rand((Hkv, f, E)).to(dtype).to(dtype) for f in frags]
    values = [torch.rand((Hkv, f, Ev)).to(dtype).to(dtype) for f in frags]
    frags = torch.tensor(frags, dtype=torch.int32)
    locations = torch.arange(0, S)[None, None, :] + (frags[:, None, None] - S)
    locations = torch.tile(locations, (1, W, 1)).to(torch.int32)

    expected, qk, att = hogwild_sdpa(queries, locations, keys, values, scale=scale)

    # ok, write test bin

    with open(file_name, "wb") as file:
        file.write(header.tobytes())
        file.write(frags.numpy().tobytes())
        file.write(queries.to(dtype).view(torch.uint8).numpy().tobytes())
        file.write(locations.numpy().tobytes())
        for f in range(F):
            file.write(keys[f].to(dtype).view(torch.uint8).numpy().tobytes())
        for f in range(F):
            file.write(values[f].to(dtype).view(torch.uint8).numpy().tobytes())
        file.write(expected.numpy().tobytes())
        file.write(qk.numpy().tobytes())
        file.write(att.numpy().tobytes())


generate_test_data("bench.bin", 4, 2, 40, 8, 128, 128, 1, [20000, 10, 15, 10], dtype="bf16")
#generate_test_data("test.bin", 4, 2, 40, 8, 128, 128, 2, [200, 10, 15, 10])
#generate_test_data("test_prefill.bin", 1, 1, 40, 8, 128, 128, 5000, [5000])
#generate_test_data("test.bin", 1, 1, 40, 8, 128, 128, 10, [100])
#generate_test_data("test.bin", 1, 1, 1, 1, 128, 128, 1, [100])
