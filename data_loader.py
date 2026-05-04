import os
import pandas as pd

def load_all_datasets(data_folder="data"):
    frames = []

    def add(text_series, label_series, source):
        df = pd.DataFrame({"text": text_series, "label": label_series, "source": source})
        df = df.dropna(subset=["text", "label"])
        frames.append(df)
        print(f"  [LOAD] {source}: {len(df)} rows")

    # 1. 2020-12-31-DynamicallyGeneratedHateDataset
    try:
        df = pd.read_csv(f"{data_folder}/2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv")
        # label: 'hate' or 'nothate'
        add(df["text"], (df["label"] == "hate").astype(int), "dynamically_generated_hate")
    except Exception as e: print(f"  [SKIP] dynamically_generated_hate: {e}")

    # 2. aggression_parsed_dataset
    try:
        df = pd.read_csv(f"{data_folder}/aggression_parsed_dataset.csv")
        # oh_label: 1=aggression, 0=not
        add(df["Text"], df["oh_label"], "aggression")
    except Exception as e: print(f"  [SKIP] aggression: {e}")

    # 3. attack_parsed_dataset
    try:
        df = pd.read_csv(f"{data_folder}/attack_parsed_dataset.csv")
        add(df["Text"], df["oh_label"], "attack")
    except Exception as e: print(f"  [SKIP] attack: {e}")

    # 4. cyberbullying_dataset
    try:
        df = pd.read_csv(f"{data_folder}/cyberbullying_dataset.csv")
        # label: 1=bullying, 0=not
        add(df["text"], df["label"], "cyberbullying_dataset")
    except Exception as e: print(f"  [SKIP] cyberbullying_dataset: {e}")

    # 5. cyberbullying_tweets
    try:
        df = pd.read_csv(f"{data_folder}/cyberbullying_tweets.csv")
        # cyberbullying_type: 'not_cyberbullying' or other categories
        label = (df["cyberbullying_type"] != "not_cyberbullying").astype(int)
        add(df["tweet_text"], label, "cyberbullying_tweets")
    except Exception as e: print(f"  [SKIP] cyberbullying_tweets: {e}")

    # 6. data.csv — tab-separated, skip (malformed headers)
    try:
        df = pd.read_csv(f"{data_folder}/data.csv", sep="\t", header=None, names=["text","label","extra"])
        label = (df["label"] != "none").astype(int)
        add(df["text"], label, "data_tsv")
    except Exception as e: print(f"  [SKIP] data_tsv: {e}")

    # 7. dataset (2).csv
    try:
        df = pd.read_csv(f"{data_folder}/dataset (2).csv")
        # Insult: 1=insult, 0=not
        add(df["Comment"], df["Insult"], "dataset_2")
    except Exception as e: print(f"  [SKIP] dataset_2: {e}")

    # 8. en_dataset
    try:
        df = pd.read_csv(f"{data_folder}/en_dataset.csv")
        # sentiment: 'hateful', 'abusive', 'normal', 'spam'
        label = (df["sentiment"].isin(["hateful", "abusive"])).astype(int)
        add(df["tweet"], label, "en_dataset")
    except Exception as e: print(f"  [SKIP] en_dataset: {e}")

    # 9. en_dataset_with_stop_words
    try:
        df = pd.read_csv(f"{data_folder}/en_dataset_with_stop_words.csv")
        label = (df["sentiment"].isin(["hateful", "abusive"])).astype(int)
        add(df["tweet"], label, "en_dataset_stopwords")
    except Exception as e: print(f"  [SKIP] en_dataset_stopwords: {e}")

    # 10. final_dataset_with_txt
    try:
        df = pd.read_csv(f"{data_folder}/final_dataset_with_txt.csv")
        add(df["text"], df["label"], "final_dataset")
    except Exception as e: print(f"  [SKIP] final_dataset: {e}")

    # 11. final_hateXplain
    try:
        df = pd.read_csv(f"{data_folder}/final_hateXplain.csv")
        # label: 'hatespeech','offensive','normal'
        label = (df["label"].isin(["hatespeech", "offensive"])).astype(int)
        add(df["comment"], label, "final_hateXplain")
    except Exception as e: print(f"  [SKIP] final_hateXplain: {e}")

    # 12. hateXplain — skip (post_tokens is tokenized, not raw text)
    print("  [SKIP] hateXplain.csv — tokens format, not usable as-is")

    # 13. kaggle_parsed_dataset
    try:
        df = pd.read_csv(f"{data_folder}/kaggle_parsed_dataset.csv")
        add(df["Text"], df["oh_label"], "kaggle_parsed")
    except Exception as e: print(f"  [SKIP] kaggle_parsed: {e}")

    # 14. labeled_data
    try:
        df = pd.read_csv(f"{data_folder}/labeled_data.csv")
        # class: 0=hate, 1=offensive, 2=neither
        label = (df["class"].isin([0, 1])).astype(int)
        add(df["tweet"], label, "labeled_data")
    except Exception as e: print(f"  [SKIP] labeled_data: {e}")

    # 15. labeled_data_of_cyberbullying
    try:
        df = pd.read_csv(f"{data_folder}/labeled_data_of_cyberbullying.csv")
        label = (df["class"].isin([0, 1])).astype(int)
        add(df["tweet"], label, "labeled_data_cyberbullying")
    except Exception as e: print(f"  [SKIP] labeled_data_cyberbullying: {e}")

    # 16. toxicity_parsed_dataset
    try:
        df = pd.read_csv(f"{data_folder}/toxicity_parsed_dataset.csv")
        add(df["Text"], df["oh_label"], "toxicity_parsed")
    except Exception as e: print(f"  [SKIP] toxicity_parsed: {e}")

    # 17. train.csv — multi-label toxicity
    try:
        df = pd.read_csv(f"{data_folder}/train.csv")
        # bullying if any toxic column is 1
        label = (df[["malignant","highly_malignant","rude","threat","abuse","loathe"]].max(axis=1))
        add(df["comment_text"], label, "train_toxicity")
    except Exception as e: print(f"  [SKIP] train_toxicity: {e}")

    # 18. train1.csv
    try:
        df = pd.read_csv(f"{data_folder}/train1.csv")
        add(df["tweet"], df["label"], "train1")
    except Exception as e: print(f"  [SKIP] train1: {e}")

    # 19. train_E6oV3lV.csv
    try:
        df = pd.read_csv(f"{data_folder}/train_E6oV3lV.csv")
        add(df["tweet"], df["label"], "train_E6oV3lV")
    except Exception as e: print(f"  [SKIP] train_E6oV3lV: {e}")

    # 20. twitter_parsed_dataset
    try:
        df = pd.read_csv(f"{data_folder}/twitter_parsed_dataset.csv")
        add(df["Text"], df["oh_label"], "twitter_parsed")
    except Exception as e: print(f"  [SKIP] twitter_parsed: {e}")

    # 21. twitter_racism_parsed_dataset
    try:
        df = pd.read_csv(f"{data_folder}/twitter_racism_parsed_dataset.csv")
        add(df["Text"], df["oh_label"], "twitter_racism")
    except Exception as e: print(f"  [SKIP] twitter_racism: {e}")

    # 22. twitter_sexism_parsed_dataset
    try:
        df = pd.read_csv(f"{data_folder}/twitter_sexism_parsed_dataset.csv")
        add(df["Text"], df["oh_label"], "twitter_sexism")
    except Exception as e: print(f"  [SKIP] twitter_sexism: {e}")

    # 23. ultimate_cyberbullying_dataset
    try:
        df = pd.read_csv(f"{data_folder}/ultimate_cyberbullying_dataset.csv")
        add(df["text"], df["label"], "ultimate_cyberbullying")
    except Exception as e: print(f"  [SKIP] ultimate_cyberbullying: {e}")

    # 24. youtube_parsed_dataset
    try:
        df = pd.read_csv(f"{data_folder}/youtube_parsed_dataset.csv")
        add(df["Text"], df["oh_label"], "youtube_parsed")
    except Exception as e: print(f"  [SKIP] youtube_parsed: {e}")
    
    # 25. final_complete_dataset
    try:
        df = pd.read_csv(f"{data_folder}/final_complete_dataset.csv")
        add(df["text"], df["label"], "final_complete_dataset")
    except Exception as e: print(f"  [SKIP] final_complete_dataset: {e}")
    # Combine all
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"])
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce")
    combined = combined.dropna(subset=["label"])
    combined["label"] = combined["label"].astype(int)
    print(f"\n✅ Total: {len(combined)} rows")
    print(f"   Bullying=1: {combined['label'].sum()} | Safe=0: {(combined['label']==0).sum()}")
    # Clamp any label > 1 to 1 (treat 2,3 etc as bullying)
    combined["label"] = combined["label"].clip(0, 1)
    return combined


if __name__ == "__main__":
    df = load_all_datasets()
    print(df.head())