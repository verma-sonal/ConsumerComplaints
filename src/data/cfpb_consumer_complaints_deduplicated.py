import pandas as pd
import os

# --- Configuration ---
input_file = r'/content/drive/MyDrive/consumercomplaints/raw/complaints.csv'  # replace with your file path
output_file = r'/content/drive/MyDrive/consumercomplaints/raw/deduplicated_output.csv'
chunk_size = 50_000  # adjust based on your memory limits
dedup_column = 'Consumer complaint narrative'  # replace with the column you want to deduplicate on

# --- Process ---
seen = set()
first_chunk = True

# Delete output file if it already exists
if os.path.exists(output_file):
    os.remove(output_file)

for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False):
    # Only keep rows where the column value has not been seen before
    mask = ~chunk[dedup_column].isin(seen)
    deduped_chunk = chunk[mask]
    
    # Update seen values
    seen.update(deduped_chunk[dedup_column].tolist())

    if not deduped_chunk.empty:
        # Write immediately in small batches
        deduped_chunk.to_csv(output_file, index=False, mode='a', header=first_chunk)
        first_chunk = False

print(f"\n✅ Deduplication complete! Output saved as: {output_file}")