import os
import shutil
import urllib.request
import pandas as pd

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

    # Assign labels
    true_df["label"] = "REAL"
    fake_df["label"] = "FAKE"

    # Select only required columns
    true_df = true_df[["text", "label"]]
    fake_df = fake_df[["text", "label"]]

    # Drop any null or empty texts
    true_df = true_df.dropna(subset=["text"])
    fake_df = fake_df.dropna(subset=["text"])

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
    # Clean up temporary downloads
    print("Cleaning up temporary CSV files...")
    if os.path.exists(TRUE_TEMP):
        os.remove(TRUE_TEMP)
    if os.path.exists(FAKE_TEMP):
        os.remove(FAKE_TEMP)
    print("Cleanup complete.")
