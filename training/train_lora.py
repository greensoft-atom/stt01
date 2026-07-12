"""LoRA fine-tune of Whisper on a corpus built by build_corpus.py.

Run on a CUDA GPU machine (a few hours of rented GPU is enough). On this CPU-only
dev box use --smoke to validate the whole script end-to-end with whisper-tiny.

    python training\\train_lora.py --corpus data\\corpus --model openai/whisper-small
    python training\\train_lora.py --smoke                       # CPU validation run

Deliberate design: NO generation-based eval during training (PEFT + generate is
fragile). Train on loss only; the adoption gate is eval_cer.py on the held-out
split AFTER merging and converting — that measures the real production setup.

After training, the script prints the exact CTranslate2 conversion command that
produces a model folder the pipeline can load.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000


class CorpusDataset:
    """metadata.csv + wav clips -> Whisper input_features/labels. No `datasets` lib
    (its Audio decoding needs torchcodec, unnecessary here)."""

    def __init__(self, split_dir: Path, processor):
        self.dir = split_dir
        self.processor = processor
        with (split_dir / "metadata.csv").open(encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if r["transcription"].strip()]
        if not self.rows:
            raise SystemExit(f"no usable rows in {split_dir}/metadata.csv")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import soundfile as sf

        row = self.rows[idx]
        audio, sr = sf.read(self.dir / row["file_name"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == SAMPLE_RATE, f"{row['file_name']}: expected 16 kHz, got {sr}"
        features = self.processor(audio, sampling_rate=SAMPLE_RATE).input_features[0]
        labels = self.processor.tokenizer(row["transcription"]).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class Collator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, batch):
        inputs = [{"input_features": b["input_features"]} for b in batch]
        out = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = [{"input_ids": b["labels"]} for b in batch]
        padded = self.processor.tokenizer.pad(labels, return_tensors="pt")
        ids = padded["input_ids"].masked_fill(padded.attention_mask.ne(1), -100)
        # Labels begin with <|startoftranscript|> (the DECODER START token, not the
        # tokenizer's bos token); strip it — the model re-prepends it when shifting.
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]
        out["labels"] = ids
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "corpus")
    parser.add_argument("--model", default="openai/whisper-small",
                        help="Base HF model (small first; large-v3-turbo if small shows gains)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dir (default: training/output/<model>-court-lora)")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--smoke", action="store_true",
                        help="CPU validation: whisper-tiny, 2 steps, no CUDA required")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    if args.smoke:
        args.model = "openai/whisper-tiny"
        args.batch = 2
    elif not torch.cuda.is_available():
        print("ERROR: no CUDA GPU found. Real training needs a GPU machine "
              "(see docs/FINETUNING_ROADMAP.md). Use --smoke to validate the "
              "script on CPU.")
        return 1

    out_dir = args.out or (PROJECT_ROOT / "training" / "output" /
                           (args.model.split("/")[-1] + "-court-lora"))

    processor = WhisperProcessor.from_pretrained(args.model, language="ko", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    ))
    model.print_trainable_parameters()

    train_ds = CorpusDataset(args.corpus / "train", processor)
    print(f"train clips: {len(train_ds)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=max(1, 32 // args.batch),
        learning_rate=args.lr,
        warmup_steps=200 if not args.smoke else 0,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        # smoke: no checkpoints — this dev box's antivirus breaks mid-train saves
        save_strategy="no" if args.smoke else "epoch",
        report_to=[],
        max_steps=2 if args.smoke else -1,
        dataloader_num_workers=0,
        remove_unused_columns=False,   # required: our Dataset is not an HF Dataset
        label_names=["labels"],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds,
        data_collator=Collator(processor, model.config.decoder_start_token_id),
    )
    trainer.train()

    merged_dir = out_dir / "merged"
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    # transformers v5 writes processor_config.json, but ct2-transformers-converter
    # needs the classic preprocessor_config.json — save the feature extractor too.
    processor.feature_extractor.save_pretrained(merged_dir)
    print(f"\nMerged model saved: {merged_dir}")
    print("\nNEXT STEPS:")
    print(f"  1. Convert for the pipeline:\n"
          f"     ct2-transformers-converter --model \"{merged_dir}\" "
          f"--output_dir \"{PROJECT_ROOT / 'models' / (out_dir.name + '-ct2')}\" "
          f"--quantization int8 --copy_files tokenizer.json preprocessor_config.json")
    print(f"  2. Gate on held-out CER:\n"
          f"     python training\\eval_cer.py --model \"{PROJECT_ROOT / 'models' / (out_dir.name + '-ct2')}\" "
          f"--corpus \"{args.corpus / 'eval'}\"")
    print("  3. Adopt only if CER beats the current production model on the same split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
