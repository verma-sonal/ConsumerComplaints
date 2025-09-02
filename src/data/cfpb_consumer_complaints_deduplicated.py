import pandas as pd
import os

# --- Configuration ---
input_file = r'/content/drive/MyDrive/consumercomplaints/raw/complaints.csv'
output_file = r'/content/drive/MyDrive/consumercomplaints/raw/deduplicated_output.csv'
chunk_size = 100_000
dedup_column = 'Consumer complaint narrative'

# --- Processing ---
first_chunk = True
seen = set()

# Remove output file if it already exists
if os.path.exists(output_file):
    os.remove(output_file)

for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False):
    # Drop rows with NaN in deduplication column
    chunk = chunk.dropna(subset=[dedup_column])

    # Drop duplicate narratives already seen
    chunk = chunk[~chunk[dedup_column].isin(seen)]

    # Update seen values (only the new unique ones)
    seen.update(chunk[dedup_column])

    # Write to file
    if not chunk.empty:
        chunk.to_csv(output_file, index=False, mode='a', header=first_chunk)
        first_chunk = False

print(f"\nDone! Deduplicated output saved to: {output_file}")
