# BERT-FT Baseline

Clean BERT fine-tuning baseline for the RAE-HMC dataset.

This baseline uses:

- BERT encoder
- Linear sigmoid classification head
- BCEWithLogitsLoss
- Validation threshold tuning

It intentionally excludes hierarchy-specific modules, retrieval memory, contrastive losses, and path loss.
