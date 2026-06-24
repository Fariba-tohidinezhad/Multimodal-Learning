# Cross-Attention Multimodal Learning for Predicting Response to Neoadjuvant Imatinib in GIST

## Overview

This repository contains the code for developing explainable multimodal deep learning models that predict response to neoadjuvant imatinib in gastrointestinal stromal tumors (GIST) using:

* 3D CT imaging
* Clinical variables
* Transformer-based multimodal fusion
* Self-supervised pretraining
* LoRA fine-tuning
* SMAC3 hyperparameter optimization

The framework is implemented using **FuseMedML**, **PyTorch Lightning**, and **SMAC3**.

---

## Study Design

<p align="center">
  <img src="figures/figure1_study_design.png" width="900">
</p>

*Multicenter cohort design for MAE pretraining and response prediction.*

---

## Framework Overview

<p align="center">
  <img src="figures/figure2_framework.png" width="900">
</p>

*Self-supervised MAE pretraining, LoRA adaptation, and multimodal cross-attention architecture.*

---

## Implemented Models

### Clinical Model

Transformer-based model using clinical variables only.

### Imaging Model

Transformer-based model using tumor-centered CT patches only.

### Cross-Attention Model

Proposed multimodal architecture integrating clinical and imaging tokens through bidirectional cross-attention.

---

## Citation

```bibtex
@article{tohidinezhad2026gist,
  title={Cross-Attention Multimodal Learning for Predicting Response to Neoadjuvant Imatinib in Gastrointestinal Stromal Tumors},
  author={Tohidinezhad, Fariba and colleagues},
  year={2026}
}
```

---

## Acknowledgements

Developed at the Biomedical Imaging Group Rotterdam (BIGR), Erasmus MC.

Supported by:

* Dutch Research Council (NWO)
* AiNed Fellowship Program
* Stichting Hanarth Fonds

Computing resources were provided by SURF and Erasmus MC.
