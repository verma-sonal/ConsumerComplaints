"""###Import libraries"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation #LDA
from google.colab import drive  # Import drive from google.colab

"""### Runtime Check"""

import torch
import time

def check_gpu_and_recommend():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("No GPU detected! You are running on CPU. RESTART and select GPU in Runtime settings.")
        return

    # Get GPU name
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")

    # Decision logic
    gpu_name_lower = gpu_name.lower()

    if "a100" in gpu_name_lower:
        print("Perfect! A100 GPU detected. Proceed with full embedding!")
    elif "l4" in gpu_name_lower:
        print("Great! L4 GPU detected. Proceed — fast and efficient for your workload!")
    elif "t4" in gpu_name_lower:
        print("Good. T4 GPU detected. Proceed, but embedding will be slower.")
    elif "p100" in gpu_name_lower:
        print("Very Good! P100 GPU detected. Proceed — fast embeddings expected.")
    elif "k80" in gpu_name_lower:
        print("Slow GPU (K80) detected. RECOMMEND: Runtime → Restart until you get a T4/L4/A100.")
    else:
        print("Unknown GPU detected. Proceed cautiously.")

    # Additional info
    print("\n Tip: Always save intermediate results if you are not on A100 or L4 to avoid losses!")

# Run the GPU check
check_gpu_and_recommend()

"""### Mount Drive and Load Data"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

dataFile = 'deduplicated_complaints_cleaned.csv'
path = '/content/drive/MyDrive/data/data/' + dataFile

# Load with safer memory handling
df = pd.read_csv(path, sep=",", low_memory=False)

#Take Backup
backup_df = df.copy()

# Print all column names
print(df.columns.tolist())  # Also prints all column names

"""### Remove rows where narratives are missing and remove redacted text like XX, XXXX"""

# Remove rows where narrative is missing
df = df.dropna(subset=["Consumer complaint narrative"]).reset_index(drop=True)

# Remove redacted text like XXXX etc
import re

# Define a clean function
def clean_placeholders(text):
    # Remove standalone XX, XXX, XXXX surrounded by word boundaries
    cleaned_text = re.sub(r'\bX{2,}\b', '', text, flags=re.IGNORECASE)
    # Replace multiple spaces caused by removal
    cleaned_text = re.sub(' +', ' ', cleaned_text).strip()
    return cleaned_text

# Apply to the complaint narratives
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(clean_placeholders)

print("Successfully removed XX/XXX/XXXX placeholders without distorting complaint meaning.")

"""###Add a column for year and group by year"""

##Add a name column year and group by year

# Convert 'Date received' to datetime
df['Date received'] = pd.to_datetime(df['Date received'], errors='coerce')

# Create a new 'year' column
df['year'] = df['Date received'].dt.year

# Group by year and count number of complaints
complaints_per_year = df.groupby('year').size().reset_index(name='Complaint_Count')

# Display
print(complaints_per_year)

"""###Filter by year 2024, one product and one issue under that and write it to CSV"""

# Filter by Year = 2024
df_2024 = df[df['year'] == 2024]

# Filter by Product and Issue
product_filter = 'Credit reporting or other personal consumer reports'
issue_filter = 'Improper use of your report'

filtered_df = df_2024[(df_2024['Product'] == product_filter) & (df_2024['Issue'] == issue_filter)]

#print the number of records
print(f"Total complaints for selected Product and Issue = {filtered_df.shape[0]}")

# Save the filtered dataset
output_path = '/content/drive/MyDrive/data/data/filtered_complaints_2024_credit_reporting_improper_use.csv'

filtered_df.to_csv(output_path, index=False)

print(f"Filtered dataset saved successfully at: {output_path}")

"""#End-to-End Complaint Narrative Clustering and Trend Analysis with External Economic Indicators"

#### Step 1: Install necessary libraries
"""

!pip install -q sentence-transformers hdbscan umap-learn scikit-learn pandas matplotlib pandas_datareader

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pandas_datareader.data as web
import datetime
import torch

from google.colab import drive
drive.mount('/content/drive')

"""#### Step 2: Check if GPU is available"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\033[1mRunning on {device.upper()}\033[0m")

"""#### Step 3: Load filtered dataset from drive & Prepare text data"""

file_path = '/content/drive/MyDrive/data/data/filtered_complaints_2024_credit_reporting_improper_use.csv'
input_df = pd.read_csv(file_path)

# Step 1: Drop missing complaint narratives from input_df itself
input_df = input_df.dropna(subset=['Consumer complaint narrative']).reset_index(drop=True)

# Step 2: Now safely create texts
print("\033[1mPreparing complaint narratives...\033[0m")
texts = input_df['Consumer complaint narrative'].tolist()

texts

"""#### Step 4: Load Embedding Model (force GPU if available) & Generate Embeddings"""

# Load Embedding Model (force GPU if available)
print("\033[1mLoading SentenceTransformer model...\033[0m")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Generate Embeddings
print("\033[1mGenerating embeddings...\033[0m")
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, device=device)

# Important: Adjust input_df length to match the number of embeddings
# input_df = input_df.iloc[:len(embeddings)].reset_index(drop=True)

"""#### Step 5: Dimensionality Reduction for clustering"""

# Dimensionality Reduction for Clustering
#
# This step reduces the high-dimensional complaint embeddings down to 5 dimensions
# using Uniform Manifold Approximation and Projection (UMAP).
#
# Purpose:
# - Simplify complex embedding vectors while preserving their structure.
# - Make clustering (e.g., HDBSCAN) faster, more accurate, and computationally efficient.
#
# Key Parameters:
# - n_components=5: Target dimensionality (5 dimensions).
# - random_state=42: Ensures reproducibility of results across runs.
#
# Output:
# - 'embeddings_umap' — a reduced-dimension representation of complaint narratives,
#   optimized for unsupervised clustering.
# Dimensionality Reduction for clustering
print("\033[1mReducing dimensions...\033[0m")
reducer = umap.UMAP(n_components=5, random_state=42)
embeddings_umap = reducer.fit_transform(embeddings)

"""#### Step 6: Clustering with HDBSCAN & add clusters to input dataframe"""

# Clustering with HDBSCAN
print("\033[1mClustering complaints...\033[0m")
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric='euclidean', prediction_data=True)
cluster_labels = clusterer.fit_predict(embeddings_umap)

# Add clusters to input DataFrame
input_df['cluster'] = cluster_labels

# Save cluster output in CSV and save the model as well in pickle file

import pickle

def save_results(input_df, clusterer, csv_path, pkl_path):
    """
    Save clustered complaint data to CSV and trained HDBSCAN model to Pickle (.pkl).

    Parameters:
    ----------
    input_df : pd.DataFrame
        The DataFrame containing complaints and assigned cluster labels.

    clusterer : object
        The trained clustering model (e.g., HDBSCAN object).

    csv_path : str
        Path to save the CSV output.

    pkl_path : str
        Path to save the Pickle (.pkl) model file.

    Returns:
    -------
    None
    """
    # Save DataFrame
    input_df.to_csv(csv_path, index=False)
    print(f"\033[1mDataFrame saved to: {csv_path}\033[0m")

    # Save model
    with open(pkl_path, 'wb') as f:
        pickle.dump(clusterer, f)
    print(f"\033[1m HDBSCAN model saved to: {pkl_path}\033[0m")


# File paths where you want to save
csv_output_path = '/content/drive/MyDrive/data/data/approach2-clustered_complaints.csv'
model_output_path = '/content/drive/MyDrive/data/data/approach2-hdbscan_clusterer.pkl'

# Save both results
save_results(input_df, clusterer, csv_output_path, model_output_path)

input_df['cluster']

"""####Create lables for these clusters using LLaMA"""

!pip install transformers accelerate torch

# Setup Authentication
from huggingface_hub import login

# Paste your Hugging Face API token here
HUGGINGFACE_TOKEN = "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"

# Authenticate
login(token=HUGGINGFACE_TOKEN)
print("Hugging Face Authentication Successful!")

#Load the LLaMA model from huggingface
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Choose LLaMA 3.2 model (modify if needed)
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16, use_auth_token=True)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("LLaMA 3.2 Model Loaded Successfully!")

#Function to label a cluster using LLaMA
def label_cluster_with_llama(sample_texts):
    prompt = (
        "You are analyzing consumer complaint narratives.\n"
        "Given the following complaints, Summarize the main issue in 3 to 8 words. Return only the short theme, not a sentence.\n\n"
    )
    for i, complaint in enumerate(sample_texts):
        prompt += f"{i+1}. {complaint.strip()}\n"
    prompt += "\nTheme:"

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the theme (after the prompt)
        if "Theme:" in generated_text:
            theme_part = generated_text.split("Theme:")[-1].strip()
            return theme_part
        else:
            return generated_text.strip()

    except Exception as e:
        print("Error:", e)
        return "Unknown"

import pandas as pd
from tqdm import tqdm

def generate_tags_batched_safe(
    texts,
    tokenizer,
    model,
    device,
    batch_size=8,
    save_every=10,
    save_path=None
):
    all_tags = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Tag Batches"):
        batch = texts[i:i+batch_size]

        # Prepare the prompt
        prompt = "Generate 4-6 short tags (separated by commas) for each complaint below:\n"
        for j, text in enumerate(batch):
            prompt += f"{j+1}. {text.strip()}\n"
        prompt += "\nTags:\n"

        try:
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            # Decode and parse
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            lines = decoded.split("Tags:")[-1].strip().split("\n")

            for j in range(len(batch)):
                try:
                    line = lines[j]
                    tags = line.split(".", 1)[-1] if "." in line else line
                    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                    all_tags.append(tag_list)
                except IndexError:
                    all_tags.append([])  # Fallback for missing lines

        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {e}")
            all_tags.extend([[] for _ in batch])  # Fallback for whole batch

        # Save checkpoint
        if save_path and ((i // batch_size + 1) % save_every == 0):
            pd.DataFrame({"Tags": all_tags}).to_csv(save_path, index=False)
            print(f"Saved checkpoint at batch {i//batch_size + 1} to {save_path}")

    return all_tags

#Function to generate tags from complaint text using LLaMA
from tqdm import tqdm

def generate_tags_batched(texts, batch_size=8):
    all_tags = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Tag Batches"):
        batch = texts[i:i+batch_size]

        # Build prompt for batch
        prompt = (
            "You are an assistant generating short, relevant tags for consumer complaints.\n"
            "Generate 2 to 3 concise tags (not sentences), separated by commas, for each complaint.\n\n"
        )
        for idx, text in enumerate(batch):
            prompt += f"{idx+1}. {text.strip()}\n"
        prompt += "\nReturn tags in the same numbered order:\n"

        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=30,  # Allow enough tokens for multiple lines
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tag_section = response.split("Return tags in the same numbered order:")[-1].strip()
            tag_lines = tag_section.split("\n")

            # Clean up and map lines to each item
            parsed_batch = []
            for j in range(len(batch)):
                try:
                    line = tag_lines[j]
                    # Extract after "1. ", "2. ", etc.
                    tags = line.split(".", 1)[-1].strip() if "." in line else line.strip()
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                    parsed_batch.append(tag_list)
                except IndexError:
                    # If fewer lines than expected, fill in with empty list
                    parsed_batch.append([])

            all_tags.extend(parsed_batch)

        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {e}")
            all_tags.extend([[] for _ in batch])  # Fill empty for failed batch

    return all_tags

import time
from tqdm import tqdm

tqdm.pandas()  # Enables progress bar for pandas apply

cluster_labels_LLaMA = {}

drive.mount('/content/drive')

# Load the saved clustered CSV
final_df = pd.read_csv('/content/drive/MyDrive/data/data/approach2-clustered_complaints.csv')


# Generate tags from each complaint narrative using LLaMA
print("Generating tags in batches...")

texts = final_df['Consumer complaint narrative'].tolist()
tag_results = generate_tags_batched_safe(
    texts,
    tokenizer=tokenizer,
    model=model,
    device=device,
    batch_size=12,   # Try 12 or 16 if your GPU has enough memory
    save_every=10,
    save_path="/content/partial_tags_2.csv"
)

assert len(tag_results) == len(final_df), "Mismatch between tag results and complaints!"
final_df['Tags'] = tag_results

print("Tags generated successfully!")

print("Total rows in final_df:", len(final_df))
print("Tags generated:", len(tag_results))

# Generate Theme for cluster using LLaMA
for cluster_id in sorted(final_df["cluster"].unique()):
    if cluster_id == -1:
        continue

    sample_texts = final_df[final_df["cluster"] == cluster_id]["Consumer complaint narrative"].dropna()
    if len(sample_texts) == 0:
        cluster_labels_LLaMA[cluster_id] = "Unknown"
        print(f"Cluster {cluster_id}: No data.")
        continue

    sample_texts = sample_texts.sample(min(5, len(sample_texts))).tolist()
    print(f"Sending cluster {cluster_id} to LLaMA 3.2...")

    label = label_cluster_with_llama(sample_texts)
    cluster_labels_LLaMA[cluster_id] = label
    print(f"Cluster {cluster_id} label: {label}\n")

    #time.sleep(6)

#Update the final_df by adding additional column to store labels generated by LLaMA. Save it in CSV
final_df["cluster_label_LLaMA"] = final_df["cluster"].map(cluster_labels_LLaMA)
final_df["cluster_label_LLaMA"] = final_df["cluster_label_LLaMA"].fillna("Noise/Outlier")

final_df.to_csv("/content/drive/MyDrive/data/data/approach2_LLaMA_Labels.csv", index=False)

df_sample = pd.read_csv("/content/drive/MyDrive/data/data/approach2_LLaMA_Labels.csv", nrows=1000)  # Load first 1000 rows
df_sample.head(100)

"""#### Step 7: Basic Cluster Analysis"""

# Mount Drive and Load final DataFrame if needed
from google.colab import drive
import pandas as pd

drive.mount('/content/drive')
final_df = pd.read_csv('/content/drive/MyDrive/data/data/approach2_LLaMA_Labels.csv')

# Basic Cluster Analysis
num_clusters = len(set(final_df["cluster"])) - (1 if -1 in final_df["cluster"].values else 0)
print(f"\033[1mNumber of clusters found: {num_clusters}\033[0m")

# View Unique Cluster Labels (Generated by LLaMA)
print("\033[1mViewing LLaMA Cluster Labels...\033[0m")
unique_llama_labels = final_df['cluster_label_LLaMA'].unique()

print(f"\033[1mTotal unique LLaMA labels: {len(unique_llama_labels)}\033[0m")
print(unique_llama_labels)

# Group complaints by year and month
print("\033[1mAggregating complaint counts per month...\033[0m")
final_df['Date received'] = pd.to_datetime(final_df['Date received'], errors='coerce')
final_df['year_month'] = final_df['Date received'].dt.to_period('M')

# Complaint trend over time
trend_df = final_df.groupby(['year_month']).size().to_frame(name='Complaint_Count')

# Ready trend_df for plotting or analysis
print("\033[1mTrend data prepared!\033[0m")

"""####Step 8: Fetch External Economic Indicators from FRED, merge with complaints and save to CSV"""

# Fetch External Economic Indicators from FRED
print("\033[1mFetching external factors from FRED API...\033[0m")
start = datetime.datetime(2024, 1, 1)
end = datetime.datetime(2024, 12, 31)

inflation = web.DataReader('CPIAUCSL', 'fred', start, end).resample('M').mean()
unemployment = web.DataReader('UNRATE', 'fred', start, end).resample('M').mean()
mortgage_rate = web.DataReader('MORTGAGE30US', 'fred', start, end).resample('M').mean()

external_factors = pd.concat([inflation, unemployment, mortgage_rate], axis=1)
external_factors.columns = ['Inflation', 'Unemployment', 'MortgageRate']

# Merge complaint trends with external factors
print("\033[1mMerging complaint trends with external factors...\033[0m")
trend_df.index = trend_df.index.to_timestamp()
final_df = trend_df.merge(external_factors, left_index=True, right_index=True, how='inner')

# Save merged output to CSV
output_path = '/content/drive/MyDrive/data/data/complaints_external_analysis.csv'
final_df.to_csv(output_path)
print(f"\033[1mMerged data saved to {output_path}\033[0m")

"""#### Step 9: Visualize trends"""

# Visualize trends
print("\033[1mVisualizing complaint counts and external factors...\033[0m")
fig, ax1 = plt.subplots(figsize=(14,8))

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Complaint Count', color=color)
ax1.plot(final_df.index, final_df['Complaint_Count'], color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('External Factors', color=color)
ax2.plot(final_df.index, final_df['Inflation'], color='red', linestyle='--', label='Inflation')
ax2.plot(final_df.index, final_df['Unemployment'], color='orange', linestyle='--', label='Unemployment')
ax2.plot(final_df.index, final_df['MortgageRate'], color='green', linestyle='--', label='Mortgage Rate')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.legend(loc='upper center')
plt.title('Complaint Counts vs Economic Indicators (2024)')
plt.show()

print("\033[1mEnd-to-end pipeline completed successfully!\033[0m")