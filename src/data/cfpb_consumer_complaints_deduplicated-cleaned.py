import pandas as pd
import re

# Path to input file
input_path = r'/content/drive/MyDrive/consumercomplaints/raw/deduplicated_output.csv'
output_path = r'/content/drive/MyDrive/consumercomplaints/raw/deduplicated_complaints_cleaned.csv'

# Load CSV
print("Loading data...")
df = pd.read_csv(input_path)

# Function to clean redacted text (e.g., XXXX, XX/XX/XXXX)
def clean_redacted(text):
    if pd.isna(text):
        return text
    text = re.sub(r'X{2,}/X{2,}/X{2,}', '', text)  # Remove XX/XX/XXXX-style dates
    text = re.sub(r'X{2,}', '', text)              # Remove XXXX-like blocks
    text = re.sub(r'\s+', ' ', text).strip()       # Normalize whitespace
    return text

# Apply cleaning
print("Cleaning redacted text...")
df['Cleaned_Complaint'] = df['Consumer complaint narrative'].astype(str).apply(clean_redacted)

# Overwrite the narrative with the cleaned version where available
# (keeps original if cleaned is NaN or empty)
mask = df['Cleaned_Complaint'].notna() & (df['Cleaned_Complaint'].str.len() > 0)
df.loc[mask, 'Consumer complaint narrative'] = df.loc[mask, 'Cleaned_Complaint']

# Drop helper column
df.drop(columns=['Cleaned_Complaint'], inplace=True)

# Save to new file
df.to_csv(output_path, index=False)
print(f"Cleaned file saved to: {output_path}")
