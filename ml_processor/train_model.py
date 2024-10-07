import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Function to load model and tokenizer
def load_model_and_tokenizer(model_name="bert-base-uncased", num_labels=2):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

# Function to load dataset from CSV file
def load_dataset(csv_file_path):
    df = pd.read_csv(csv_file_path)
    dataset = Dataset.from_pandas(df)
    return dataset

# Function to tokenize dataset
def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Function to split dataset into train and test sets
def split_dataset(tokenized_dataset, test_size=0.2):
    split_datasets = tokenized_dataset.train_test_split(test_size=test_size)
    return split_datasets['train'], split_datasets['test']


# Function to create trainer
def create_trainer(model, train_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_dir='./logs',    
        logging_steps=10,              
        save_steps=500,                 
        eval_strategy="steps",  
        eval_steps=500  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    return trainer


# Function to save model and tokenizer
def save_model_and_tokenizer(model, tokenizer, output_dir="./saved_model"):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train():
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load and tokenize the dataset
    csv_file_path = "data/training_data.csv"
    dataset = load_dataset(csv_file_path)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    train_dataset, test_dataset = split_dataset(tokenized_dataset)

    trainer = create_trainer(model, train_dataset, test_dataset)

    trainer.train()

    results = trainer.evaluate()
    print("Evaluation results:", results)

    # Save the trained model and tokenizer
    save_model_and_tokenizer(model, tokenizer)

if __name__ == "__main__":
    train()
