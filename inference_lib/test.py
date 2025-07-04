from typing import Tuple

import torch
import math
import numpy as np


torch.ops.load_library("libhogatt.so")


def hogwild_sdpa_ref(queries: torch.Tensor, locations: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor],
                     scale: float) -> torch.Tensor:
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
        att.masked_fill_(mask, float("-inf"))
        qk.append(att)
        vals.append(val)

    qk = torch.concat(qk, dim=-1)
    vals = torch.concat(vals, dim=-2).to(torch.float32)
    att = torch.softmax(scale * qk, dim=-1)
    result = att @ vals
    return result


def hogwild_sdpa(queries: torch.Tensor, locations: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor],
                 scale: float, fragment_lengths=None) -> torch.Tensor:
    out = torch.empty((queries.size(1), queries.size(2), queries.size(3), values[0].size(3)), dtype=torch.float32, device=queries.device)
    if fragment_lengths is None:
        fragment_lengths = torch.tensor([k.size(2) for k in keys], dtype=torch.int32, device=queries.device)
    keys = [k.to(torch.float32).squeeze(0).contiguous() for k in keys]
    values = [v.to(torch.float32).squeeze(0).contiguous() for v in values]
    torch.ops.libhogatt.hogwild_sdpa(out, scale, locations.to(torch.int32), queries.to(torch.float32).contiguous(), fragment_lengths, keys, values)
    return out.to(queries.dtype)


@torch.no_grad()
def test_custom_kernel(F: int, W: int, Hq: int, Hkv: int, E: int, Ev: int, S: int, # noqa
                       frags: list[int], scale: float = None):
    # TODO make input distributions more interesting
    torch.random.manual_seed(42)
    if scale is None:
        scale = 1.0 / math.sqrt(E)
    queries = torch.rand((F, W, Hq, S, E))
    keys = [torch.rand((Hkv, f, E)) for f in frags]
    values = [torch.rand((Hkv, f, Ev)) for f in frags]
    frags = torch.tensor(frags, dtype=torch.int32)
    locations = torch.arange(0, S)[None, None, :] + (frags[:, None, None] - S)
    locations = torch.tile(locations, (1, W, 1)).to(torch.int32)

    expected = hogwild_sdpa_ref(queries, locations, keys, values, scale=scale)
    keys = [k[None, ...].to("cuda") for k in keys]
    values = [k[None, ...].to("cuda") for k in values]
    actual = hogwild_sdpa(queries.to("cuda"), locations.to("cuda"), keys, values, scale=scale)
    print(expected)
    print(expected.cpu() - actual.cpu())


#test_custom_kernel(4, 2, 40, 8, 128, 128, 2, [200, 10, 15, 10])
test_custom_kernel(1, 1, 1, 1, 128, 128, 1, [100])
