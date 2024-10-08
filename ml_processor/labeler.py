import random
import pandas as pd

from .split import split_into_sentences



def read_and_label_file(file_path, label):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        sentences = split_into_sentences(text)
        labeled_data = [{"text": sentence, "label": label} for sentence in sentences]
    return labeled_data

def save_dataset(data, save_path):
    if data:
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
    else:
        print("No data to save!")

def build_training_data(novel_file_path = "ml_data/novel-like.txt", non_novel_file_path = "ml_data/non-novel.txt", save_path = "ml_data/training_data.csv"):
    novel_data = read_and_label_file(novel_file_path, label=0)
    non_novel_data = read_and_label_file(non_novel_file_path, label=1)

    combined_data = novel_data + non_novel_data
    random.shuffle(combined_data)

    save_dataset(combined_data, save_path)

if __name__ == "__main__":
    build_training_data()
