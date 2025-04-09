<a href='https://arxiv.org/abs/2504.06261'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
<a href='https://eqimp.github.io/hogwild_llm><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;

#  Hogwild! Inference: Parallel LLM Generation with a Concurrent Attention Cache 

Official PyTorch implementation for 

`Hogwild! Inference: Parallel LLM Generation with a Concurrent Attention Cache`

---

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


Simple example with minimal prompt: [__`basic_example.ipynb`__](./basic_example.ipynb)

Hogwild! Inference with full prompt: [__`full_example.ipynb`__](./full_example.ipynb)

## Cite

If you found this work useful, please consider citing:

```
@misc{rodionov2025hogwildinferenceparallelllm,
      title={Hogwild! Inference: Parallel LLM Generation via Concurrent Attention}, 
      author={Gleb Rodionov and Roman Garipov and Alina Shutova and George Yakushev and Vage Egiazarian and Anton Sinitsin and Denis Kuznedelev and Dan Alistarh},
      year={2025},
      eprint={2504.06261},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.06261}, 
}
```
