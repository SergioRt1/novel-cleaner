import torch
import torch.nn.functional as F
import os
import time
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from tkinter import filedialog


from .split import split_into_sentences

def load_model():
    print('GPU acceleration: ', torch.cuda.is_available())  # Should return True if CUDA/ROCm is properly configured.
    print('GPU device is: ', torch.cuda.get_device_name())

    # Load the saved model and tokenizer
    model_path = "./saved_model" 
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Ensure the model is in evaluation mode
    model.eval()
    model.to('cuda')

    return model, tokenizer

results_folder = "./results_prediction"

# Function to batch evaluate the sentences
def evaluate_novel(novel_text, model, tokenizer, batch_size=16):
    sentences = split_into_sentences(novel_text)
    print(f"Text split in {len(sentences)} sentences")

    results = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Evaluating Sentences"):
   
        batch = sentences[i:i+batch_size]
        
        # Tokenize the batch
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Run the model on the batch
        with torch.no_grad():  # Disable gradient calculation for evaluation
            outputs = model(**inputs)
        

        logits = outputs.logits
        
        # Calculate probabilities using softmax
        probs = F.softmax(logits, dim=-1)
        prediction_certainties, predictions = torch.max(probs, dim=-1)

        # Convert to lists
        predictions = predictions.tolist()
        prediction_certainties = prediction_certainties.tolist()

        # Append sentence, prediction, and certainty for each sentence in the batch
        batch_results = list(zip(batch, predictions, prediction_certainties))
        results.extend(batch_results)
    
    return results

# Function to read the novel from a text file
def read_novel_from_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read(), os.path.basename(file_path).split('.')[0]  # return text and file name without extension
    return "", ""

# Function to save the results to a CSV file in a 'results' folder
def save_results_to_csv(df, base_filename):
    os.makedirs(results_folder, exist_ok=True)
    save_path = os.path.join(results_folder, f"{base_filename}_predictions.csv")
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

# Function to manually review and filter the novel content
def save_filtered_novel(df, original_text, base_filename, modification_handler):
    filtered_sentences = []
    
    # Loop through the sentences and review flagged sentences (Prediction == 1)
    for index, row in df.iterrows():
        sentence = row['Sentence']
        prediction = row['Prediction']
        certainty = row['Certainty']
        
        # If the sentence is flagged as non-novel content, review it
        if prediction == 1:
            edited_sentence, keep = modification_handler(sentence, certainty)
            if keep:
                filtered_sentences.append(edited_sentence)
        else:
            filtered_sentences.append(sentence)
    
    # Reconstruct the novel by joining the filtered sentences
    filtered_novel = ''.join(filtered_sentences)

    # Save the filtered novel to a text file in the 'results' folder
    
    os.makedirs(results_folder, exist_ok=True)
    filtered_novel_path = os.path.join(results_folder, f"{base_filename}_filtered.txt")
    
    with open(filtered_novel_path, "w", encoding="utf-8") as file:
        file.write(filtered_novel)
    
    print(f"Filtered novel saved to {filtered_novel_path}")

# Function to handle CLI interactions
def modification_handler_cli(sentence, certainty):
    print(f"Flagged as non-novel content ({certainty:.4%}):\n{sentence}")
    action = input("Choose action - (k)eep, (e)dit, (r)emove (any other): ").strip().lower()

    lines_printed = 2
    for i in range(lines_printed):
        print("\033[A\r\033[K", end='')

    if action == 'k':
        # Keep the sentence
        return sentence, True
    elif action == 'e':
        # Edit the sentence
        edited_sentence = input("Edit the sentence: ").strip()
        print("\033[A\r\033[K", end='')
        return edited_sentence, True
    # If 'r' or other, the sentence will be removed
    return "", False

def modification_handler_remove_all(sentence, certainty):
    cleaned_sentence = sentence.replace("\n", " ")
    print(f'Flagged {certainty:.4%}:{cleaned_sentence}')
    time.sleep(0.2)

    print("\033[A\r\033[K", end='')
    # The sentence will be removed
    return "", False

# Function to predict and return a DataFrame with predictions and certainties
def predict(model, tokenizer, novel_text: str):
    if novel_text:
        evaluated_results = evaluate_novel(novel_text, model, tokenizer)
        
        # Convert results to DataFrame, including certainty scores
        df = pd.DataFrame(evaluated_results, columns=["Sentence", "Prediction", "Certainty"])
        
        return df
    else:
        print("No file selected or file is empty.")
        return None

# Main execution: reading the novel from a file and evaluating it
if __name__ == "__main__":

    # Prompt the user to select a text file
    novel_text, base_filename = read_novel_from_file()
    print("Novel loaded")

    model, tokenizer = load_model()
    df = predict(model, tokenizer, novel_text)

    save_results_to_csv(df, base_filename)
    save_filtered_novel(df, novel_text, base_filename, modification_handler_remove_all)
