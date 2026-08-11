# Relation to Prior and Concurrent Work

ASMC builds on probabilistic inference and sequence-level sampling for
autoregressive language models.

## Power Sampling for LLM Reasoning

Karan and Du, [*Reasoning with Sampling: Your Base Model is Smarter Than You
Think*](https://arxiv.org/abs/2510.14901) (ICLR 2026 Oral), introduced the
sequence-level power distribution as a training-free target for LLM reasoning
and used autoregressive Metropolis-Hastings/MCMC to approximately sample it.
ASMC adopts the same objective but replaces the serial chain with a
GPU-parallel population of weighted particles, adaptive particle allocation,
and cache-coherent Transformer KV-state resampling.

## Sequential Monte Carlo for Language Models

Zhao et al., [*Probabilistic Inference in Language Models via Twisted
Sequential Monte Carlo*](https://proceedings.mlr.press/v235/zhao24c.html) (ICML
2024), developed SMC for language-model inference under unnormalized
sequence-level targets. Their learned twist functions estimate future
potentials; ASMC instead defines its internal target solely from
`p_theta(x | c)^alpha`, without an external reward or learned twist.

## Concurrent Particle Power Sampling

[Power-SMC](https://arxiv.org/abs/2602.10273) (Azizi et al., 2026) is concurrent
work that also applies SMC to the global sequence-level power distribution.
Both methods target complete-trajectory power sampling and replace serial MCMC
with GPU-parallel particles. ASMC emphasizes adaptive allocation,
cache-coherent KV/state resampling, deployment-oriented latency, and collapse
diagnostics; Power-SMC emphasizes prefix-only proposal analysis,
Rényi-entropy weight-instability analysis, and exponent-bridging proposals.
We view the methods as concurrent and complementary.

## BibTeX

```bibtex
@inproceedings{karan2026reasoning,
  title     = {Reasoning with Sampling: Your Base Model is Smarter Than You Think},
  author    = {Karan, Aayush and Du, Yilun},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  note      = {Oral presentation},
  url       = {https://arxiv.org/abs/2510.14901}
}

@inproceedings{zhao2024probabilistic,
  title     = {Probabilistic Inference in Language Models via Twisted Sequential {M}onte {C}arlo},
  author    = {Zhao, Stephen and Brekelmans, Rob and Makhzani, Alireza and Grosse, Roger Baker},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages     = {60704--60748},
  year      = {2024},
  volume    = {235},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v235/zhao24c.html}
}

@article{azizi2026powersmc,
  title   = {{Power-SMC}: Low-Latency Sequence-Level Power Sampling for Training-Free {LLM} Reasoning},
  author  = {Azizi, Seyedarmin and Baghaei Potraghloo, Erfan and Ahmadi, Minoo and Kundu, Souvik and Pedram, Massoud},
  journal = {arXiv preprint arXiv:2602.10273},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.10273}
}
```
