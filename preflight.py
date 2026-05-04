import sys
print("[ Checking Python & libraries... ]")

# 1. Check all imports
try:
    import torch
    print(f"  ✅ torch {torch.__version__}")
except: print("  ❌ torch — run: pip install torch"); sys.exit()

try:
    import transformers
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
    print(f"  ✅ transformers {transformers.__version__}")
except Exception as e: print(f"  ❌ transformers — {e}"); sys.exit()

try:
    import datasets
    from datasets import Dataset
    print(f"  ✅ datasets {datasets.__version__}")
except: print("  ❌ datasets — run: pip install datasets"); sys.exit()

try:
    import sklearn
    print(f"  ✅ scikit-learn {sklearn.__version__}")
except: print("  ❌ scikit-learn — run: pip install scikit-learn"); sys.exit()

try:
    import accelerate
    print(f"  ✅ accelerate {accelerate.__version__}")
except: print("  ❌ accelerate — run: pip install accelerate"); sys.exit()

# 2. Check TrainingArguments accepts new param names
print("\n[ Checking TrainingArguments compatibility... ]")
try:
    args = TrainingArguments(
        output_dir="tmp_test",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    print("  ✅ eval_strategy=epoch works")
except TypeError:
    try:
        args = TrainingArguments(
            output_dir="tmp_test",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
        )
        print("  ✅ evaluation_strategy=epoch works (older API)")
        print("  ⚠️  Update train_bert.py: use evaluation_strategy instead of eval_strategy")
    except Exception as e:
        print(f"  ❌ TrainingArguments broken: {e}"); sys.exit()

# 3. Check tokenizer + model download
print("\n[ Checking DistilBERT download... ]")
try:
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    print("  ✅ DistilBERT loaded OK")
except Exception as e:
    print(f"  ❌ DistilBERT failed: {e}"); sys.exit()

# 4. Test a mini training loop (10 fake samples)
print("\n[ Running mini training smoke test... ]")
try:
    import numpy as np
    texts = ["hello world"] * 10
    labels = [0, 1] * 5
    enc = tokenizer(texts, truncation=True, padding=True, max_length=32)
    enc["labels"] = labels
    tiny_dataset = Dataset.from_dict(enc)

    mini_args = TrainingArguments(
        output_dir="tmp_test",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
        use_cpu=True,
    )
    trainer = Trainer(
        model=model,
        args=mini_args,
        train_dataset=tiny_dataset,
        eval_dataset=tiny_dataset,
    )
    trainer.train()
    print("  ✅ Mini training loop works!")
except Exception as e:
    print(f"  ❌ Mini training failed: {e}"); sys.exit()

# 5. Check data loader
print("\n[ Checking data_loader... ]")
try:
    from data_loader import load_all_datasets
    df = load_all_datasets()
    assert len(df) > 100000, "Dataset too small"
    assert "text" in df.columns
    assert "label" in df.columns
    assert df["label"].isin([0, 1]).all(), "Labels not binary"
    print(f"  ✅ data_loader OK — {len(df)} rows, labels are clean 0/1")
except Exception as e:
    print(f"  ❌ data_loader failed: {e}"); sys.exit()

# Cleanup
import shutil, os
if os.path.exists("tmp_test"): shutil.rmtree("tmp_test")

print("\n🎉 ALL CHECKS PASSED — safe to run python train_bert.py")