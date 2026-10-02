# Banking77 Intent Classifier

A text classification model that routes customer support messages to one of 77 banking-related intents, fine-tuned on top of BERT.

## Overview

This project fine-tunes `bert-base-uncased` on the Banking77 dataset to classify customer messages into 77 intent categories (e.g. "card is being declined", "why was I charged a fee"). The model also includes a basic confidence-based abstention mechanism, so it can flag uncertain predictions for human review instead of guessing.

## Dataset

- **Source:** [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) (HuggingFace)
- **Classes:** 77 intents
- **Split:**
  - Train: 8,502 examples (85% of original train set)
  - Validation: 1,501 examples (15% of original train set, stratified split)
  - Test: 3,080 examples (original test set, untouched during development)

## Model

- **Backbone:** `bert-base-uncased` (~110M parameters), fully fine-tuned (not frozen)
- **Architecture:**
  - BERT encoder → CLS token representation
  - Dropout (0.25)
  - Linear layer (768 → 384)
  - Dropout
  - Linear layer (384 → 77)
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam, learning rate 2e-5
- **Epochs:** 5
- **Batch size:** 32
- **Max sequence length:** 32 tokens

## Results (Validation Set)

| Metric | Score |
|---|---|
| Macro F1 | 0.9073 |

Training and validation loss decreased steadily across all 5 epochs, with no signs of significant overfitting.

## Abstention

Instead of always returning a prediction, the model estimates its own confidence using the maximum softmax probability across the 77 classes. When confidence falls below a chosen threshold, the message is flagged for human review (`ESCALATE`) rather than risking a wrong classification.

## How to Run

1. Open the notebook in Google Colab (GPU runtime recommended).
2. Run all cells in order to load the data, build the model, and either train from scratch or load the saved weights (`intent_model.pt`).
3. Use the `predict_with_confidence()` function to classify new messages and get a confidence score.

## Limitations

- Trained and validated only on in-scope Banking77 messages; out-of-scope detection has not been separately evaluated.
- The confidence threshold for abstention has not been tuned or validated against a held-out signal.
- Evaluated on a single training run (no multi-seed variance check).
