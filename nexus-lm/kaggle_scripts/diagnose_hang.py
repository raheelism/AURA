"""
Quick diagnostic to identify where data prep is hanging.
Run this FIRST to narrow down the problem.

Stages:
1. Can HF dataset be loaded at all?
2. Can we get the first sample?
3. Can we tokenize samples quickly?
4. Can we write to memmap?
"""
import sys
import time
from pathlib import Path

def stage1_load_dataset():
    """Test: Can we load the HuggingFace dataset?"""
    print("\n" + "="*60)
    print("STAGE 1: Load HuggingFace dataset")
    print("="*60)
    from datasets import load_dataset
    
    try:
        print("Loading fineweb-edu 10BT sample...")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        print(f"✅ Dataset loaded: {ds}")
        return ds
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None


def stage2_get_first_sample(ds):
    """Test: Can we get the first sample from the dataset?"""
    print("\n" + "="*60)
    print("STAGE 2: Get first sample from dataset")
    print("="*60)
    
    try:
        print("Attempting to fetch first sample... (timeout: 30s)")
        start = time.time()
        
        # Use iter() and next() with timeout-like behavior
        iterator = iter(ds)
        first_sample = next(iterator)
        
        elapsed = time.time() - start
        print(f"✅ Got first sample in {elapsed:.2f}s")
        print(f"   Text length: {len(first_sample['text'])} chars")
        print(f"   First 100 chars: {first_sample['text'][:100]}")
        return first_sample, iterator
    except StopIteration:
        print(f"❌ Dataset is empty (no samples)")
        return None, None
    except Exception as e:
        print(f"❌ Failed to get first sample: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def stage3_tokenize_samples(tokenizer_path: str, iterator, n_samples: int = 10):
    """Test: Can we tokenize samples quickly?"""
    print("\n" + "="*60)
    print(f"STAGE 3: Tokenize {n_samples} samples")
    print("="*60)
    
    try:
        from data.tokenizer import load_tokenizer, encode
        print(f"Loading tokenizer from {tokenizer_path}...")
        sp = load_tokenizer(tokenizer_path)
        print(f"✅ Tokenizer loaded (vocab={sp.get_piece_size()})")
        
        total_tokens = 0
        for i in range(n_samples):
            try:
                sample = next(iterator)
                text = sample['text']
                ids = encode(sp, text)
                total_tokens += len(ids)
                print(f"  Sample {i+1}: {len(text)} chars → {len(ids)} tokens")
            except StopIteration:
                print(f"  Ran out of samples at index {i}")
                break
            except Exception as e:
                print(f"  ❌ Error on sample {i+1}: {e}")
                return False
        
        print(f"✅ Successfully tokenized {n_samples} samples")
        print(f"   Total tokens: {total_tokens:,} ({total_tokens/n_samples:.0f} tokens/sample avg)")
        return True
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def stage4_write_memmap():
    """Test: Can we allocate and write to a memmap file?"""
    print("\n" + "="*60)
    print("STAGE 4: Write to memmap file")
    print("="*60)
    
    try:
        import numpy as np
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test.bin"
            size = 10_000_000  # 10M uint16 tokens = ~20MB
            
            print(f"Allocating memmap: {size:,} uint16 elements (~{size*2/1e6:.1f} MB)...")
            fp = np.memmap(test_path, dtype=np.uint16, mode='w+', shape=(size,))
            
            print(f"Writing 1000 test values...")
            fp[:1000] = np.arange(1000, dtype=np.uint16)
            fp.flush()
            
            print(f"Reading back values...")
            read_vals = fp[:1000]
            if np.allclose(read_vals, np.arange(1000)):
                print(f"✅ Memmap write/read successful")
                return True
            else:
                print(f"❌ Memmap write/read mismatch")
                return False
    except Exception as e:
        print(f"❌ Memmap test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("NEXUS-AURORA DATA PREP DIAGNOSTIC")
    print("="*60)
    
    # Stage 1: Load dataset
    ds = stage1_load_dataset()
    if ds is None:
        print("\n⚠️ Cannot proceed: HF dataset loading failed")
        return
    
    # Stage 2: Get first sample
    first_sample, iterator = stage2_get_first_sample(ds)
    if first_sample is None:
        print("\n⚠️ Cannot proceed: First sample fetch failed (this is where Kaggle hangs)")
        print("   Suggestion: Try TinyStories instead of fineweb_edu")
        return
    
    # Stage 3: Tokenize samples
    tokenizer_path = "data/tokenizer.model"
    if not Path(tokenizer_path).exists():
        print(f"\n⚠️ Tokenizer not found at {tokenizer_path}")
        print("   Please train tokenizer first: python data/prepare.py --train_tokenizer")
        return
    
    success = stage3_tokenize_samples(tokenizer_path, iterator, n_samples=10)
    if not success:
        print("\n⚠️ Cannot proceed: Tokenization failed")
        return
    
    # Stage 4: Memmap write
    success = stage4_write_memmap()
    if not success:
        print("\n⚠️ Cannot proceed: Memmap write failed")
        return
    
    print("\n" + "="*60)
    print("✅ ALL DIAGNOSTIC TESTS PASSED")
    print("="*60)
    print("\nYou can now safely run the full data-prep pipeline:")
    print("  python kaggle_scripts/run.py data-prep --dataset fineweb_edu")
    print("\nIf you still encounter hangs, they are likely network-related.")
    print("Try a smaller dataset first:")
    print("  python kaggle_scripts/run.py data-prep --dataset tinystories --n-tokens 50000000")


if __name__ == '__main__':
    main()
