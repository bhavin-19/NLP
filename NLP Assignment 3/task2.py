import sys
import subprocess


#Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import torch
import re
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score
import matplotlib.pyplot as plt
import nltk

# Download NLTK data for BLEU score
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

def load_and_split_dataset(file_path):
    try:
        #Loading the dataset
        df = pd.read_csv(file_path)
        
        required_columns = ['PID', 'Social Media Post', 'Normalized Claim']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Dataset is missing one or more required columns: 'PID', 'Social Media Post', 'Normalized Claim'")
        
        #Splitting into train (70%), validation (15%), and test (15%)
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        print(f"Dataset split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

#Loading and splitting the dataset
train_df, val_df, test_df = load_and_split_dataset('CLAN-SAMPLES.csv')

def load_replacements(file_path):
    replacements = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    replacements[key.strip()] = value.strip()
        return replacements
    except FileNotFoundError:
        print(f"Error: Replacement file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading replacements: {e}")
        sys.exit(1)

#Loading replacements
replacements = load_replacements('replacements.txt')


def preprocess_text(text, replacements):
    if not isinstance(text, str):
        return ""  
    
    #Expanding the contractions and abbreviations
    for key, value in replacements.items():
        text = re.sub(r'\b' + re.escape(key) + r'\b', value, text)
    
    #Removing URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    #Removing special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    #Removing extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    #Converting to lowercase
    text = text.lower()
    return text

#Appling preprocessing to the 'Social Media Post' column
for df in [train_df, val_df, test_df]:
    df['input_text'] = df['Social Media Post'].apply(lambda x: preprocess_text(x, replacements))


def prepare_dataset_t5(df, tokenizer, max_input_length=512, max_target_length=128):
    def tokenize_function(examples):
        input_texts = ["normalize text: " + text for text in examples['input_text']]
        inputs = tokenizer(input_texts, max_length=max_input_length, truncation=True, padding='max_length')
        targets = tokenizer(examples['Normalized Claim'], max_length=max_target_length, truncation=True, padding='max_length')
        inputs['labels'] = targets['input_ids']
        return inputs
    
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

#Initializing BART tokenizer and model
try:
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
except Exception as e:
    print(f"Error loading BART model/tokenizer: {e}")
    sys.exit(1)

#Preparing the datasets
train_dataset_bart = prepare_dataset_t5(train_df, bart_tokenizer)
val_dataset_bart = prepare_dataset_t5(val_df, bart_tokenizer)

#Defining training arguments
training_args_bart = TrainingArguments(
    output_dir='./bart_results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./bart_logs',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    logging_steps=10,
)

#Initializing trainer
trainer_bart = Trainer(
    model=bart_model,
    args=training_args_bart,
    train_dataset=train_dataset_bart,
    eval_dataset=val_dataset_bart,
)

#Training BART
# try:
#     trainer_bart.train()
# except Exception as e:
#     print(f"Error training BART: {e}")
#     sys.exit(1)


#Initializing T5 tokenizer and model
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration

# Force redownload by ignoring cached files
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base', force_download=True)
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base', force_download=True)
    # t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    # t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
except Exception as e:
    print(f"Error loading T5 model/tokenizer: {e}")
    sys.exit(1)

#Preparing the datasets
train_dataset_t5 = prepare_dataset_t5(train_df, t5_tokenizer)
val_dataset_t5 = prepare_dataset_t5(val_df, t5_tokenizer)

#Defining training arguments
training_args_t5 = TrainingArguments(
    output_dir='./t5_results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./t5_logs',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    logging_steps=10,
)

#Initializing the trainer
trainer_t5 = Trainer(
    model=t5_model,
    args=training_args_t5,
    train_dataset=train_dataset_t5,
    eval_dataset=val_dataset_t5,
)

# #Training T5
# try:
#     trainer_t5.train()
# except Exception as e:
#     print(f"Error training T5: {e}")
#     sys.exit(1)


def evaluate_model(model, tokenizer, test_df):
    try:
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        predictions = []
        references = test_df['Normalized Claim'].tolist()
        
        for input_text in test_df['input_text']:
            input_ids = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).input_ids.to(device)
            output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predictions.append(pred)
        
        #ROUGE-L
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
        avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
        
        #BLEU-4
        bleu_scores = [sentence_bleu([ref.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25)) 
                       for ref, pred in zip(references, predictions)]
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        
        #BERTScore
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
        avg_bert_score = F1.mean().item()
        
        return {'ROUGE-L': avg_rouge_l, 'BLEU-4': avg_bleu, 'BERTScore': avg_bert_score}
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {'ROUGE-L': 0, 'BLEU-4': 0, 'BERTScore': 0}

# #Evaluate both models
# bart_metrics = evaluate_model(bart_model, bart_tokenizer, test_df)
# t5_metrics = evaluate_model(t5_model, t5_tokenizer, test_df)

# print("BART Metrics:", bart_metrics)
# print("T5 Metrics:", t5_metrics)

# # Select the best model (e.g., based on BERTScore)
# best_model = bart_model if bart_metrics['BERTScore'] > t5_metrics['BERTScore'] else t5_model
# best_tokenizer = bart_tokenizer if bart_metrics['BERTScore'] > t5_metrics['BERTScore'] else t5_tokenizer
# best_metrics = bart_metrics if bart_metrics['BERTScore'] > t5_metrics['BERTScore'] else t5_metrics

# #Saving the best model
# try:
#     best_model.save_pretrained('./best_model')
#     best_tokenizer.save_pretrained('./best_model')
#     print("Best model saved to './best_model'")
# except Exception as e:
#     print(f"Error saving model: {e}")

def inference(test_csv_path, model_path='./best_model'):
    try:
        #Loading the best model and tokenizer
        model = BartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)
        
        #Loading and preprocess test data
        test_df = pd.read_csv(test_csv_path)
        if 'Social Media Post' not in test_df.columns:
            raise ValueError("'Social Media Post' column missing in test.csv")
        
        test_df['input_text'] = test_df['Social Media Post'].apply(lambda x: preprocess_text(x, replacements))
        
        #Generating the predictions
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        predictions = []
        for input_text in test_df['input_text']:
            input_ids = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).input_ids.to(device)
            output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            predictions.append(pred)
        
        #Computing metrics 
        if 'Normalized Claim' in test_df.columns:
            metrics = evaluate_model(model, tokenizer, test_df)
            print("Test Metrics:", metrics)
        else:
            print("Ground truth not available in test.csv. Metrics not computed.")
        
        #Saving predictions to a file
        test_df['Generated Claim'] = predictions
        test_df.to_csv('test_predictions.csv', index=False)
        print("Predictions saved to 'test_predictions.csv'")
        
        return predictions
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return []

# Example usage (uncomment during demo)
predictions = inference('test.csv')

