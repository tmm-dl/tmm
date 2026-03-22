"""
DistilBERT text-classification model for TMM example 01.

The TMM python plugin discovers this class by scanning the module for a
torch.nn.Module subclass and instantiating it with no arguments.

Input tensors (via DLPack from the hf-tokenize preprocessor):
    inputs[0]  input_ids      int32  [batch, max_length]
    inputs[1]  attention_mask int32  [batch, max_length]
    inputs[2]  labels         int64  [batch]   (squeezed from [batch, 1])

Returns a HuggingFace SequenceClassifierOutput; the TMM python plugin reads
.loss for training/validation and .logits for downstream evaluation.
"""

import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification


class Model(nn.Module):
    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.backbone = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        # input_ids / attention_mask arrive as int32 from the collator;
        # DistilBERT requires int64 for embedding lookups.
        input_ids      = input_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.long)

        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
