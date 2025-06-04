# Spanish-to-English Translator (Transformer-based)

This project implements a **Spanish-to-English neural machine translation model** from scratch using a **custom-built Transformer architecture with cross-attention**, developed entirely in **PyTorch**. The model was trained on **50 million English-Spanish sentence pairs** and demonstrates high translation accuracy without relying on prebuilt libraries like Hugging Face or OpenNMT.

---

## Project Highlights

- **Built from scratch** Transformer model (Encoder-Decoder with cross-attention)
-  Implemented in **PyTorch**
-  Trained on **50M rows** of real-world English-to-Spanish translation data
-  Achieved **87% accuracy** on held-out validation data
-  Incorporates **7 model iterations**, each with improvements in:
  - Training efficiency
  - Dataset quality and size
  - Positional encoding
  - Architecture refinements
-  Multiple training checkpoints (v1 through v7) evaluated to incrementally improve performance

---

##  Model Hosting Notice

>  Due to GitHub's 100MB file limit, the final trained model (`GPT_translation_7.0.pt`, ~550MB) could not be uploaded to this repository.

If you'd like access to the model file or earlier checkpoints, feel free to reach out or open an issue — alternate download links (e.g., Google Drive or Hugging Face) may be provided.

---

##  Accuracy and Evaluation

- Final model (v7) reached **87% accuracy** on a balanced validation set of Spanish-to-English sentence pairs.
- Accuracy is computed via exact match and BLEU-score evaluations on unseen sentence pairs.
- Key improvements came from:
  - Better positional encoding
  - Smarter batching and padding strategies
  - Rebalanced datasets with longer average sequence lengths

---
