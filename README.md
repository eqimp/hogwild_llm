#  Hogwild! Inference: Parallel LLM Generation with a Concurrent Attention Cache 

Official PyTorch implementation for  [Hogwild! Inference: Parallel LLM Generation with a Concurrent Attention Cache](...)

## Demo
<div align="center">
  <picture>
  <img src="https://github.com/user-attachments/assets/b842e693-bdb9-46d5-acef-9cfbce42911b" width="80%">
  </picture>
  <br>
  <div align="center" width="80%">
  <em>Hogwild! Inference.</em>
  </div>
  <br>
</div>


## Inference with shared cache:

### Dependencies

Install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Run with multiple workers

To try inference described in the paper you can run jupyter notebooks from notebooks/ folder:


Contiguous layout (token-wise): [./notebooks/demo_hogwild_inference.ipynb](./notebooks/demo_hogwild_inference.ipynb)

Interleaved layout (step-wise): [./notebooks/demo_hogwild_inference_interleaved_s1like.ipynb](./notebooks/demo_hogwild_inference.ipynb)

## Cite

If you found this work useful, please consider citing:

```
@misc{
}
```
