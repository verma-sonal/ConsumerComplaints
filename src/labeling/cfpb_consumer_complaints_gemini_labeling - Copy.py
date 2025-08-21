
# Install necessary libraries
import os
import re
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# ------------------------
# Configuration
# ------------------------

API_KEY = "YOUR API KEY"  # 🔁 Replace with your actual Gemini API key
INPUT_CSV = "/content/drive/MyDrive/consumercomplaints/raw/deduplicated_complaints_cleaned.csv"
CHECKPOINT_CSV = "/content/drive/MyDrive/consumercomplaints/partial_theme_checkpoint_gemini.csv"
FINAL_OUTPUT_CSV = "/content/drive/MyDrive/consumercomplaints/Gemini_Labels_theme.csv"
MAX_RETRIES = 3
SAVE_EVERY = 10

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

response = model.generate_content("List 3 types of consumer financial complaints.")
print(response.text)

### Mount Drive and Load Data
from google.colab import drive
drive.mount('/content/drive')

# functions that calls Gemini API's to get the themes/labels
def clean_theme_output(theme_text):
    if not isinstance(theme_text, str):
        return ""

    cleaned = re.sub(r"[*_`]+", "", theme_text)
    cleaned = re.split(r'[.:\n]', cleaned)[0].strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    for bad_start in [
        "you are an expert", "complaint", "note", "please", "let me know",
        "theme label", "respond"
    ]:
        if cleaned.lower().startswith(bad_start):
            return ""

    words = cleaned.split()
    if len(words) > 10:
        cleaned = " ".join(words[:7])
    return cleaned.strip()

BAD_THEMES = {"", "complaint", "dispute", "issue", "problem", "concern"}

def final_theme_filter(theme):
    return theme if theme.lower().strip() not in BAD_THEMES else ""

def generate_single_theme_gemini(complaint_text, retries=MAX_RETRIES):
    prompt_template = (
        "You are an expert analyst. Read the consumer complaint below and generate a short, concise theme label (not a full sentence). "
        "Your output must be 3 to 7 words only, like a title. Do not explain anything. "
        "Examples: Identity Theft, FDCPA Violation, Inaccurate Credit Report, Unauthorized Credit Pull.\n\n"
        "Complaint:\n{}\n\nTheme Label:"
    )
    prompt = prompt_template.format(complaint_text.strip())

    for _ in range(retries):
        try:
            response = model.generate_content(prompt)
            raw_output = response.text.strip()
            cleaned = clean_theme_output(raw_output)
            final = final_theme_filter(cleaned)
            if final:
                return final
        except Exception:
            continue
    return ""

# ------------------------
# Load Input CSV
# ------------------------

df = pd.read_csv(INPUT_CSV)

#df = pd.read_csv(INPUT_CSV, nrows=10)

df['Consumer complaint narrative'] = df['Consumer complaint narrative'].fillna("")

if os.path.exists(CHECKPOINT_CSV):
    checkpoint_df = pd.read_csv(CHECKPOINT_CSV, index_col=0)
    print(f"✅ Checkpoint found: {len(checkpoint_df)} records already processed.")
else:
    checkpoint_df = pd.DataFrame(columns=["Gemini_Themes"])

done_indices = set(checkpoint_df.index.astype(int))
print(f"🔁 Will skip {len(done_indices)} rows and resume from where left off.")

df

# ------------------------
# Process Each Complaint
# ------------------------

tqdm.pandas()

for idx in tqdm(df.index):
    if idx in done_indices:
        continue
    complaint_text = df.at[idx, 'Consumer complaint narrative']
    theme = generate_single_theme_gemini(complaint_text)
    checkpoint_df.loc[idx] = [theme]

    # Save checkpoint every 10 rows
    if idx % 10 == 0:
        checkpoint_df.to_csv(CHECKPOINT_CSV)
        print(f"Checkpoint saved at index {idx}")

# ------------------------
# Merge Results and Save Final Output
# ------------------------

df["Gemini_Themes"] = checkpoint_df["Gemini_Themes"]
df.to_csv(FINAL_OUTPUT_CSV , index=False)
print(f"\n✅ Final results saved to: {FINAL_OUTPUT_CSV}")