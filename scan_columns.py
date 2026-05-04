import os
import pandas as pd

data_folder = "data"

# Files already loaded — skip these
already_loaded = {
    "2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv",
    "aggression_parsed_dataset.csv", "attack_parsed_dataset.csv",
    "cyberbullying_dataset.csv", "cyberbullying_tweets.csv",
    "data.csv", "dataset (2).csv", "en_dataset.csv",
    "en_dataset_with_stop_words.csv", "final_dataset_with_txt.csv",
    "final_hateXplain.csv", "hateXplain.csv", "kaggle_parsed_dataset.csv",
    "labeled_data.csv", "labeled_data_of_cyberbullying.csv",
    "toxicity_parsed_dataset.csv", "train.csv", "train1.csv",
    "train_E6oV3lV.csv", "twitter_parsed_dataset.csv",
    "twitter_racism_parsed_dataset.csv", "twitter_sexism_parsed_dataset.csv",
    "ultimate_cyberbullying_dataset.csv", "youtube_parsed_dataset.csv"
}

for filename in os.listdir(data_folder):
    if filename.endswith(".csv") and filename not in already_loaded:
        filepath = os.path.join(data_folder, filename)
        try:
            df = pd.read_csv(filepath, nrows=2)
            print(f"\n✅ NEW: {filename}")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"\n❌ {filename} — ERROR: {e}")