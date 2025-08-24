
# Install necessary libraries
import os
import re
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# ------------------------
# Configuration
# ------------------------

API_KEY = "YOUR KEY"  # 🔁 Replace with your actual Gemini API key
INPUT_CSV = "/content/drive/MyDrive/consumercomplaints/raw/deduplicated_complaints_cleaned.csv"
CHECKPOINT_CSV = "/content/drive/MyDrive/consumercomplaints/partial_tags_checkpoint_gemini.csv"
FINAL_OUTPUT_CSV = "/content/drive/MyDrive/consumercomplaints/Gemini_Labels_theme_tags.csv"
MAX_RETRIES = 3
SAVE_EVERY = 10

# Read the top 10 rows from the CSV
df = pd.read_csv(INPUT_CSV, nrows=10)
df

# ------------------------
# Setup Gemini API
# ------------------------

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-pro")

!pip install -q google-generativeai

!pip install -U google-generativeai

import google.generativeai as genai

# STEP 1: Configure with API Key
genai.configure(api_key=API_KEY)  # Replace with your real key

# STEP 2: List all available models & what they support
for model in genai.list_models():
    print(f"{model.name} | supports generateContent: {'generateContent' in model.supported_generation_methods}")

# Testing connection
import google.generativeai as genai

# Replace with your actual API key
genai.configure(api_key=API_KEY)

# Basic test
model = genai.GenerativeModel("models/gemini-1.5-pro")

### Mount Drive and Load Data
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import re
from tqdm import tqdm

# Legal Pattern Matching
LEGAL_PATTERNS = [
    re.compile(r"\bFair Credit Reporting Act\b", re.IGNORECASE),
    re.compile(r"\bFair Debt Collection Practices Act\b", re.IGNORECASE),
    re.compile(r"\bFDCPA\b", re.IGNORECASE),
    re.compile(r"\bFCRA\b", re.IGNORECASE),
    re.compile(r"\b15\s*U\.?S\.?C\.?\s*§?\s*\d+[a-zA-Z\-]*\b", re.IGNORECASE),
    re.compile(r"\bSection\s+\d+[a-zA-Z\-]*\s+of\s+the\s+FCRA\b", re.IGNORECASE),
    re.compile(r"\bFlorida state law\b", re.IGNORECASE),
    re.compile(r"\bstatute of limitations\b", re.IGNORECASE),
]

def extract_statute_tags(text):
    matches = set()
    for pattern in LEGAL_PATTERNS:
        matches.update(map(str.strip, pattern.findall(text)))
    return list(matches)

# Buzzword Matching
BUZZWORDS = [
    "traffick", "fraud", "identity theft", "credit report error", "unauthorized charges",
    "FCRA violation", "FDCPA violation", "child abuse", "scam", "sexual abuse", "data breach",
    "harassment", "discrimination", "unauthorized access", "debt collection", "overcharged",
    "unlawful practices", "forgery", "abduction", "missing child", "molestation", "foreclosure"
]

buzzword_patterns = {
    word: re.compile(rf"\b{re.escape(word.split()[0])}\w*\b", re.IGNORECASE)
    for word in BUZZWORDS
}

def smart_inject_tags(text, existing_tags):
    lowered_text = text.lower()
    existing_lower = {tag.lower() for tag in existing_tags}
    injected = []

    for base, pattern in buzzword_patterns.items():
        if pattern.search(lowered_text) and base not in existing_lower:
            if base == "traffick":
                injected.append("trafficking")
            elif base in {"child", "sexual", "missing"}:
                continue
            else:
                injected.append(base)
    return list(set(existing_tags + injected))

# Gemini Tag Generation
def generate_tags_batched_gemini(texts, batch_size=4, save_every=10, save_path=None):
    all_tags = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Tagging Complaints (Gemini API)"):
        batch = texts[i:i + batch_size]

        prompt = (
            "You are a legal assistant trained to extract concise keyword-style tags from consumer complaints.\n"
            "Provide 2 to 3 concise tags per complaint. No hashtags or full sentences.\n\n"
        )
        for j, text in enumerate(batch):
            prompt += f"{j+1}. {text.strip()}\n"
        prompt += "\nTags:\n"

        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 256,
                }
            )
            tags_block = response.text.strip().split("Tags:")[-1].strip()
            lines = tags_block.strip().split('\n')

            for j in range(len(batch)):
                try:
                    line = lines[j]
                    match = re.match(r'^\d+\.\s*(.+)', line)
                    if match:
                        cleaned = match.group(1).strip()
                        if 1 <= len(cleaned.split()) <= 6:
                            tags = [tag.strip() for tag in cleaned.split(',') if tag.strip()]
                        else:
                            tags = []
                    else:
                        tags = []

                    tags = smart_inject_tags(batch[j], tags)
                    tags += extract_statute_tags(batch[j])
                    all_tags.append(list(set(tags)))

                except IndexError:
                    all_tags.append([])

        except Exception as e:
            print(f"Batch {i // batch_size + 1} failed: {e}")
            all_tags.extend([[] for _ in batch])

        if save_path and ((i // batch_size + 1) % save_every == 0):
            pd.DataFrame({"Tags": all_tags}).to_csv(save_path, index=False)
            print(f"Saved checkpoint at batch {i // batch_size + 1} to {save_path}")

    return all_tags

TEXT_COLUMN = "Consumer complaint narrative"
OUTPUT_COLUMN = "Gemini_Tags"
BATCH_SIZE = 1
SAVE_EVERY = 50  # ← save every 50

# ------------------------
# Load Data (limit to top 100 rows)
# ------------------------

# #df = pd.read_csv(INPUT_CSV)
df = pd.read_csv(INPUT_CSV).head(50)  # ← only pick top 100 rows

if os.path.exists(CHECKPOINT_CSV):
    checkpoint_df = pd.read_csv(CHECKPOINT_CSV, index_col=0)
    checkpoint_df = checkpoint_df.reindex(df.index)

    # Ensure column exists and has correct dtype
    if OUTPUT_COLUMN not in checkpoint_df.columns:
        checkpoint_df[OUTPUT_COLUMN] = pd.Series(dtype="string")
    else:
        checkpoint_df[OUTPUT_COLUMN] = checkpoint_df[OUTPUT_COLUMN].astype("string")

    processed_count = checkpoint_df[OUTPUT_COLUMN].notna().sum()
    print(f"✅ Resuming from checkpoint: {processed_count}/{len(df)} rows already processed for current 100-row subset.")
else:
    checkpoint_df = pd.DataFrame(index=df.index, columns=[OUTPUT_COLUMN])

done_indices = checkpoint_df[OUTPUT_COLUMN].dropna().index

# ------------------------
# Batch Buffering + Processing
# ------------------------

buffer = []
buffer_indices = []
completed = 0

for idx in tqdm(df.index, desc="Buffering rows for Gemini"):
    if idx in done_indices:
        continue

    complaint_text = df.at[idx, TEXT_COLUMN]
    # Skip empty/NaN narratives gracefully
    if pd.isna(complaint_text) or not str(complaint_text).strip():
        continue

    buffer.append(complaint_text)
    buffer_indices.append(idx)

    if len(buffer) == BATCH_SIZE:
        try:
            tag_batch = generate_tags_batched_gemini(buffer, batch_size=BATCH_SIZE)
            for bi, tags in zip(buffer_indices, tag_batch):
                checkpoint_df.at[bi, OUTPUT_COLUMN] = ', '.join(tags)
            completed += len(buffer)
        except Exception as e:
            print(f"❌ Failed batch at indices {buffer_indices}: {e}")

        buffer = []
        buffer_indices = []

        if completed % SAVE_EVERY == 0:
            checkpoint_df.to_csv(CHECKPOINT_CSV)
            print(f"💾 Checkpoint saved after {completed} new rows")

if buffer:
    try:
        tag_batch = generate_tags_batched_gemini(buffer, batch_size=len(buffer))
        for bi, tags in zip(buffer_indices, tag_batch):
            checkpoint_df.at[bi, OUTPUT_COLUMN] = ', '.join(tags)
    except Exception as e:
        print(f"❌ Failed final small batch: {e}")

# Final save
checkpoint_df.to_csv(CHECKPOINT_CSV)
df[OUTPUT_COLUMN] = checkpoint_df[OUTPUT_COLUMN]
df.to_csv(FINAL_OUTPUT_CSV, index=False)
print(f"\n✅ Final results saved to: {FINAL_OUTPUT_CSV}")