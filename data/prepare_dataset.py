import os
import shutil
import urllib.request
import re
import pandas as pd

def clean_news_text(text):
    if not isinstance(text, str):
        return ""
    # Strip headers like "WASHINGTON (Reuters) - " or "LONDON (Reuters) - " at the start
    text = re.sub(r'^[A-Z\s,]+ \((Reuters|REUTERS)\) -\s*', '', text)
    # Strip other standard agency headers or general location headers e.g. "WASHINGTON - "
    text = re.sub(r'^[A-Z\s,]+ -\s*', '', text)
    # Strip any leading "(Reuters) -" or "Reuters -"
    text = re.sub(r'^\s*\(?(Reuters|REUTERS)\)?\s*-\s*', '', text)
    return text.strip()

# Paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
BACKUP_PATH = os.path.join(DATA_DIR, "dataset_old.csv")

# 1. Back up the original dataset
if os.path.exists(DATASET_PATH):
    print(f"Backing up {DATASET_PATH} to {BACKUP_PATH}...")
    shutil.copyfile(DATASET_PATH, BACKUP_PATH)
    print("Backup complete.")
else:
    print("No existing dataset.csv found to back up.")

# URLs for Clément Bisaillon's dataset
TRUE_URL = "https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectFN/data/true.csv"
FAKE_URL = "https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectFN/data/fake.csv"

# Temporary download paths
TRUE_TEMP = os.path.join(DATA_DIR, "true_temp.csv")
FAKE_TEMP = os.path.join(DATA_DIR, "fake_temp.csv")

try:
    # 2. Download files
    print(f"Downloading True dataset from: {TRUE_URL}")
    urllib.request.urlretrieve(TRUE_URL, TRUE_TEMP)
    print("True dataset downloaded successfully.")

    print(f"Downloading Fake dataset from: {FAKE_URL}")
    urllib.request.urlretrieve(FAKE_URL, FAKE_TEMP)
    print("Fake dataset downloaded successfully.")

    # 3. Load datasets
    print("Loading datasets into Pandas...")
    true_df = pd.read_csv(TRUE_TEMP)
    fake_df = pd.read_csv(FAKE_TEMP)

    print(f"Original True dataset size: {len(true_df)} rows")
    print(f"Original Fake dataset size: {len(fake_df)} rows")

    # 4. Clean and select text column
    # Ensure they have 'text' column
    if "text" not in true_df.columns or "text" not in fake_df.columns:
        raise ValueError("Missing 'text' column in one of the downloaded datasets.")

    # Apply text cleaning
    print("Cleaning text columns (removing headers and formatting artifacts)...")
    true_df["text"] = true_df["text"].apply(clean_news_text)
    fake_df["text"] = fake_df["text"].apply(clean_news_text)

    # Assign labels
    true_df["label"] = "REAL"
    fake_df["label"] = "FAKE"

    # Select only required columns
    true_df = true_df[["text", "label"]]
    fake_df = fake_df[["text", "label"]]

    # Drop any null or empty texts
    true_df = true_df.dropna(subset=["text"])
    fake_df = fake_df.dropna(subset=["text"])

    # Remove rows where cleaning left them empty
    true_df = true_df[true_df["text"] != ""]
    fake_df = fake_df[fake_df["text"] != ""]

    # 5. Subsample to balanced 10,000 rows (5,000 from each class)
    SAMPLE_SIZE_PER_CLASS = 5000
    
    print(f"Subsampling {SAMPLE_SIZE_PER_CLASS} rows from each class...")
    true_sample = true_df.sample(n=min(SAMPLE_SIZE_PER_CLASS, len(true_df)), random_state=42)
    fake_sample = fake_df.sample(n=min(SAMPLE_SIZE_PER_CLASS, len(fake_df)), random_state=42)

    # Combine and shuffle
    combined_df = pd.concat([true_sample, fake_sample], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Final merged dataset size: {len(combined_df)} rows")
    print(f"Class counts:\n{combined_df['label'].value_counts()}")

    # 6. Save merged dataset
    print(f"Saving merged dataset to {DATASET_PATH}...")
    combined_df.to_csv(DATASET_PATH, index=False)
    print("Successfully saved dataset.")

    finally:
        # Delete dataframes and run GC to ensure file handles are released
        try:
            if 'true_df' in locals():
                del true_df
            if 'fake_df' in locals():
                del fake_df
            import gc
            gc.collect()
        except Exception:
            pass

        # Clean up temporary downloads
        print("Cleaning up temporary CSV files...")
        for temp_file in [TRUE_TEMP, FAKE_TEMP]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"Successfully removed {os.path.basename(temp_file)}")
                except Exception as e:
                    print(f"Warning: Could not remove {os.path.basename(temp_file)}: {e}")
        print("Cleanup complete.")
