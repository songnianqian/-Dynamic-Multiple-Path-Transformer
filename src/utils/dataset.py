# MIT License
#
# Copyright (c) 2025 Songnian Qian
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
import re

UNK_PAT = re.compile(r'(?:<unk>|<UNK>|\[UNK\]|\[unk\])')
PTB_DASH_PAT = re.compile(r'\s*@-@\s*')  # "U @-@ 40" -> "U-40"

# Moses artifacts and headings
MOSES_REPL = {
    " @,@ ": ",",
    " @.@ ": ".",
    " @-@ ": "-",      # ← add this
    " -LRB- ": "(",
    " -RRB- ": ")",
    " -LSB- ": "[",
    " -RSB- ": "]",
    " -LCB- ": "{",
    " -RCB- ": "}",
    " = = =": "===",
}
HEADING_RE = re.compile(r'^\s*=+\s.*\s=+\s*$')

def normalize_encoding(s: str) -> str:
    if "Ã" in s or "Â" in s:
        try:
            s = s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            pass
    return s

def detok_moses(s: str) -> str:
    for k, v in MOSES_REPL.items():
        s = s.replace(k, v)
    return s

class WikiTextDataset(Dataset):
    """WikiText dataset loader with fixed target creation"""

    def __init__(self, data_dir, tokenizer, max_length=128, split="train",
             random_start=True, max_samples=None, start_position=0, window_size=1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.random_start = random_start
        self.start_position = start_position
        self.window_size = window_size

        # Ensure pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"🔧 Set pad_token_id to eos_token_id: {self.tokenizer.pad_token_id}")

        print(f"📚 Loading WikiText-103 {split} data from {data_dir}...")
        print(f"🔧 Using window_size: {window_size}, max_length: {max_length}")

        data_path = Path(data_dir)
        target_file = data_path / f"wiki.{split}.txt"
        if not target_file.exists():
            raise FileNotFoundError(f"WikiText file not found: {target_file}")

        print(f"📖 Loading from: {target_file}")
        self._load_from_file(target_file, max_samples)

        # ❌ Do NOT reorder or slice self.texts
        # if self.random_start and len(self.texts) > 1000: self._apply_random_start()  # REMOVE
        # if start_position > 0: self.texts = self.texts[start_position:]             # REMOVE

        # ✅ Compute an offset instead
        total = len(self.texts)
        if total == 0:
            raise RuntimeError("Dataset loaded 0 chunks.")

        self.offset = 0
        if self.random_start and total > 1000:
            start_range = int(total * 0.1)
            end_range   = int(total * 0.9)
            self.offset = random.randint(start_range, end_range)
            print(f"🎲 RANDOM START offset = {self.offset}/{total}")

        if start_position > 0:
            self.offset = (self.offset + start_position) % total
            print(f"⏩ start_position offset applied, total offset = {self.offset}")

        # Keep optional cap AFTER load; this just reduces epoch size if you use it
        if max_samples is not None:
            self.texts = self.texts[:max_samples]

        print(f"✅ Loaded {len(self.texts)} text chunks (offset={self.offset})")

    def _apply_random_start(self):
        """Start training from a random position in the dataset"""
        total_texts = len(self.texts)
        start_range = int(total_texts * 0.1)
        end_range = int(total_texts * 0.9)
        random_start_idx = random.randint(start_range, end_range)
        
        original_length = len(self.texts)
        self.texts = self.texts[random_start_idx:] + self.texts[:random_start_idx]
        
        print(f"🎲 RANDOM START: Training will begin from position {random_start_idx}/{original_length}")
    
    def _load_from_file(self, file_path, max_samples):
        """Load and process WikiText file"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="strict") as f:
                content = f.read()

            lines = content.split('\n')
            current_chunk = ""
            chunks_created = 0

            for line in lines:
                line = line.strip()
                if not line or len(line) < 20:
                    continue

                # drop pure heading lines like "= = = Title = = ="
                if HEADING_RE.match(line):
                    continue

                # fix mojibake + Moses tokens before chunking
                line = detok_moses(normalize_encoding(line))

                # avoid building a temp string every time
                prospective_len = len(current_chunk) + len(line) + 1  # +1 for space
                if prospective_len < 600:
                    current_chunk += line + " "
                else:
                    if current_chunk.strip():
                        self.texts.append(current_chunk.strip())
                        chunks_created += 1
                        if max_samples is not None and chunks_created >= max_samples:
                            break
                    current_chunk = line + " "
                    if max_samples is not None and chunks_created >= max_samples:
                        break

            if current_chunk.strip() and (max_samples is None or chunks_created < max_samples):
                self.texts.append(current_chunk.strip())
                chunks_created += 1

        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            raise
    
    def __len__(self):
        return len(self.texts)
    
    @staticmethod
    def clean_text(s: str) -> str:
        s = normalize_encoding(s)
        s = detok_moses(s)
        s = UNK_PAT.sub(' ', s)           # remove explicit unknown placeholders
        s = PTB_DASH_PAT.sub('-', s)      # "U @-@ 40" -> "U-40"
        s = s.replace('�', ' ')           # drop replacement char
        s = re.sub(r'\s+', ' ', s).strip()
        return s


    def __getitem__(self, idx):
        real_idx = (self.offset + idx) % len(self.texts)
        text = self.clean_text(self.texts[real_idx])
        true_global_id = real_idx

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt',
            add_special_tokens=False
        )

        input_ids = encoding['input_ids'].squeeze(0)            # [L], long
        attention_mask = encoding['attention_mask'].squeeze(0)  # [L], 0/1 long
        attention_mask = attention_mask.to(torch.bool)          # ✅ make it boolean

        # Ensure we have enough tokens for context + target
        min_required_length = self.window_size + 1

        if input_ids.size(0) < min_required_length:
            print(f"⚠️ Text too short ({input_ids.size(0)} tokens), minimum needed: {min_required_length}")
            repeat_factor = (min_required_length // input_ids.size(0)) + 1
            input_ids = input_ids.repeat(repeat_factor)[:min_required_length]
            attention_mask = attention_mask.repeat(repeat_factor)[:min_required_length]  # stays bool ✅

        actual_length = input_ids.size(0)

        if actual_length >= self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            context_length = self.max_length - self.window_size
        else:
            context_length = max(1, actual_length - self.window_size)

        # Extract target BEFORE padding
        if actual_length >= context_length + self.window_size:
            target = input_ids[context_length:context_length + self.window_size]
        else:
            available_for_target = max(0, actual_length - context_length)
            if available_for_target > 0:
                target = input_ids[context_length:context_length + available_for_target]
                pad_needed = self.window_size - available_for_target
                target = torch.cat([
                    target,
                    torch.full((pad_needed,), self.tokenizer.pad_token_id, dtype=torch.long)
                ])
            else:
                target = input_ids[:self.window_size] if input_ids.size(0) >= self.window_size else input_ids
                if target.size(0) < self.window_size:
                    pad_needed = self.window_size - target.size(0)
                    target = torch.cat([
                        target,
                        torch.full((pad_needed,), self.tokenizer.pad_token_id, dtype=torch.long)
                    ])

        if torch.all(target == self.tokenizer.pad_token_id):
            print(f"⚠️ All target tokens are padding! Creating mixed target...")
            common_tokens = [262, 290, 257, 284, 837, 262, 290, 257]
            target = torch.tensor(common_tokens[:self.window_size], dtype=torch.long)

        # Pad inputs/mask to max_length if needed
        if input_ids.size(0) < self.max_length:
            pad_len = self.max_length - input_ids.size(0)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_len, dtype=torch.bool)          # ✅ pad with False (bool)
            ])

        # Final validation
        assert target.size(0) == self.window_size, f"Target size {target.size(0)} != window_size {self.window_size}"
        assert input_ids.size(0) == self.max_length, f"Input size {input_ids.size(0)} != max_length {self.max_length}"
        assert attention_mask.dtype == torch.bool, "attention_mask must be bool"  # ✅

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,   # ✅ bool
            'text': text,
            'global_id': true_global_id,
            'target': target
        }

# ADDITIONAL: Add this debug function to test your dataset
def debug_dataset_creation():
    """Test function to debug dataset creation"""
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test with a simple text
    test_texts = [
        "The quick brown fox jumps over the lazy dog. This is a test sentence.",
        "In the beginning was the Word, and the Word was with God.",
        "Machine learning is a subset of artificial intelligence."
    ]
    
    print("🧪 Testing dataset with sample texts:")
    
    # Create a minimal dataset class for testing
    class TestDataset:
        def __init__(self, texts, tokenizer, max_length=64, window_size=8):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.window_size = window_size
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            # Use the same logic as WikiTextDataset
            text = self.texts[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                return_tensors='pt',
                add_special_tokens=False
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            # Extract target before padding
            context_length = max(1, input_ids.size(0) - self.window_size)
            target = input_ids[context_length:context_length + self.window_size]
            
            # Pad target if needed
            if target.size(0) < self.window_size:
                pad_needed = self.window_size - target.size(0)
                target = torch.cat([
                    target,
                    torch.full((pad_needed,), self.tokenizer.pad_token_id, dtype=torch.long)
                ])
            
            # Pad input to max_length
            if input_ids.size(0) < self.max_length:
                pad_len = self.max_length - input_ids.size(0)
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_len, dtype=torch.long)
                ])
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'target': target,
                'text': text
            }
    
    test_dataset = TestDataset(test_texts, tokenizer)
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        print(f"\n   Test sample {i}:")
        print(f"   ├── Text: '{sample['text']}'")
        print(f"   ├── Input tokens: {sample['input_ids'][:20].tolist()}...")
        print(f"   ├── Target tokens: {sample['target'].tolist()}")
        
        try:
            target_text = tokenizer.decode(sample['target'], skip_special_tokens=False)
            print(f"   └── Target text: '{target_text}'")
        except:
            print(f"   └── Target decode failed")

# Uncomment to test:
# debug_dataset_creation()