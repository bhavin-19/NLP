

# 3 Task: Multimodal Sarcasm Explanation (MuSE) 45 Marks
# 3.1 Task Overview
# The objective of this task is to develop a Multimodal Sarcasm Explanation (MuSE) model. For a
# deeper understanding of the task and methodology, refer to the following paper: Research Paper
# 3.2 Dataset Description
# The task utilizes the MORE+ dataset, which comprises sarcastic posts sourced from social media platforms
# such as Twitter, Instagram, and Tumblr. Each sample consists of:
# • An image
# • A corresponding textual caption
# • A manually annotated sarcasm explanation
# • A manually annotated sarcasm target
# To simplify implementation, a preprocessed dataset will be provided, including:
# • All images in the dataset
# • Train, and validation pickle files containing:
# – Image descriptions (e.g. : D train.pkl)
# – Detected objects (e.g. : O train.pkl)
# • Data files (e.g. : train df.tsv) with fields: post ID (pid), text, explanation, and sarcasm target
# Access the dataset here: MORE+ Dataset

# 4

# 3.3 Model Requirements
# Students must implement the TURBO model without using Knowledge Graphs (KG) and Graph
# Convolutional Networks (GCN). The model should:
# • Extract high-level image features using a Vision Transformer
# • Concatenate token sequences following Section 4.3 of the referenced paper
# • Incorporate the sarcasm target into the explanation generation process
# • Be based on the BART base model
# • Implement a Shared Fusion Mechanism
# 3.4 Key Considerations
# 1. Ensure that the model effectively focuses on relevant tokens within the shared fusion mechanism.
# 2. The weights of the shared fusion mechanism should remain trainable.
# 3.5 Evaluation Metrics
# The model’s performance will be assessed using the following metrics, as specified in the paper:
# • ROUGE (R-L, R-1, R-2)
# • BLEU (B-1, B-2, B-3, B-4)
# • METEOR
# • BERTScore
# 3.6 Testing and Model Inference
# A test.tsv file, formatted similarly to train df.tsv, along with D test.pkl and O test.pkl files, will be
# provided for evaluation. Students are required to implement a function that:
# • Loads the trained model.
# • Processes these files.
# • Generates and stores sarcasm explanations in an output file.
# • Computes and reports all evaluation metrics.


import os
import logging
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import json
from datetime import datetime
from torchvision import transforms

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BartTokenizer, 
    BartForConditionalGeneration,
    BartConfig,
    ViTFeatureExtractor, 
    ViTModel,
    get_linear_schedule_with_warmup
)

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import bert_score
import nltk
nltk.download('punkt')

# Configuration dictionary for the model and training parameters. 
# This includes paths for saving models, results, and reports, as well as hyperparameters like learning rate, batch size, etc.
CONFIG = {
    'image_dir': "images",
    'batch_size': 8,
    'learning_rate': 3e-5,
    'num_epochs': 10,
    'max_text_length': 512,
    'max_target_length': 64,
    'model_save_path': 'checkpoints_v1',
    'results_path': 'new_results',
    'report_path': 'new_reports',
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'num_beams': 5,
    'repetition_penalty': 2.5,
    'no_repeat_ngram_size': 3,
    'length_penalty': 1.5
}

# Setup logging and reporting for berret hNDLEING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('muse_training.log'),
        logging.StreamHandler()
    ]
)

# Create directories
os.makedirs(CONFIG['model_save_path'], exist_ok=True)
os.makedirs(CONFIG['results_path'], exist_ok=True)
os.makedirs(CONFIG['report_path'], exist_ok=True)

# Initialize report 
report = {
    "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config": CONFIG,
    "training": {
        "epochs": []
    },
    "validation_results": [],
    "sample_predictions": []
}

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
report["device"] = str(device)

class MuSEDataset(Dataset):
    def __init__(self, df, desc_dict, obj_dict, tokenizer, feature_extractor, image_dir):
        self.df = df
        self.desc_dict = desc_dict
        self.obj_dict = obj_dict
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_dir = image_dir
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['pid']
        text = row['text']
        explanation = row['explanation']
        target = row['target_of_sarcasm']

        caption = self.desc_dict.get(pid, '')
        objects = ' '.join(self.obj_dict.get(pid, []))
        input_text = f"Explain sarcasm: Text: {text} | Target: {target} | Description: {caption} | Objects: {objects}"

        tokenized = self.tokenizer(
            input_text,
            max_length=CONFIG['max_text_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target = self.tokenizer(
            explanation,
            max_length=CONFIG['max_target_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Load and preprocess image
        image_path = os.path.join(self.image_dir, f"{pid}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.feature_extractor(images=image, return_tensors="pt")
        except:
            image_input = self.feature_extractor(images=Image.new("RGB", (224, 224)), return_tensors="pt")

        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': target['input_ids'].squeeze(0),
            'pixel_values': image_input['pixel_values'].squeeze(0)
        }

class TURBOMuSE(nn.Module):
    def __init__(self):
        super(TURBOMuSE, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.linear_proj = nn.Linear(self.vit.config.hidden_size, self.bart.config.d_model)
        
        # Shared fusion components
        self.fusion_layer = nn.Linear(self.bart.config.d_model * 2, self.bart.config.d_model)
        self.gate = nn.Linear(self.bart.config.d_model * 2, self.bart.config.d_model)

    def forward(self, input_ids, attention_mask, pixel_values=None, labels=None):
        # Extract vision embeddings
        if pixel_values is not None:
            vision_outputs = self.vit(pixel_values=pixel_values)
            vision_embeds = self.linear_proj(vision_outputs.last_hidden_state.mean(dim=1))
            
        # Get BART encoder outputs
        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = encoder_outputs.last_hidden_state.mean(dim=1)
        
        # Shared fusion mechanism
        if pixel_values is not None:
            combined = torch.cat([text_embeds, vision_embeds], dim=-1)
            fused = torch.sigmoid(self.gate(combined)) * text_embeds + (1 - torch.sigmoid(self.gate(combined))) * vision_embeds
            fused = self.fusion_layer(combined)
            
            # Expand fused features to match sequence length
            fused_features = fused.unsqueeze(1).expand(-1, encoder_outputs.last_hidden_state.size(1), -1)
            # Combine with original encoder outputs
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state + fused_features

        # Run through decoder if labels are provided
        if labels is not None:
            outputs = self.bart(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                encoder_outputs=encoder_outputs,
                return_dict=True
            )
            return outputs
        else:
            return encoder_outputs

def calculate_metrics(predictions, references):
    results = {}
    
    # ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = defaultdict(list)
    
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        for key in scores:
            rouge_scores[key].append(scores[key].fmeasure)
    
    for key in rouge_scores:
        results[f"rouge_{key}"] = np.mean(rouge_scores[key])
    
    # BLEU Scores
    smoothie = SmoothingFunction().method4
    references_bleu = [[ref.split()] for ref in references]
    predictions_bleu = [pred.split() for pred in predictions]
    
    for n in range(1, 5):
        try:
            bleu_score = corpus_bleu(
                references_bleu,
                predictions_bleu,
                weights=tuple([1.0/n]*n),
                smoothing_function=smoothie
            )
            results[f"bleu_{n}"] = bleu_score
        except:
            results[f"bleu_{n}"] = 0.0
    
    # METEOR Score
    meteor_scores = []
    for ref, pred in zip(references, predictions):
        try:
            score = meteor_score([ref.split()], pred.split())
            meteor_scores.append(score)
        except:
            meteor_scores.append(0.0)
    results["meteor"] = np.mean(meteor_scores)
    
    # BERTScore
    P, R, F1 = bert_score.score(predictions, references, lang='en')
    results["bertscore_precision"] = P.mean().item()
    results["bertscore_recall"] = R.mean().item()
    results["bertscore_f1"] = F1.mean().item()
    
    return results



# 3.5 Evaluation Metrics
# The model’s performance will be assessed using the following metrics, as specified in the paper:
# • ROUGE (R-L, R-1, R-2)
# • BLEU (B-1, B-2, B-3, B-4)
# • METEOR
# • BERTScore
def evaluate_model(model, dataloader, device, tokenizer, epoch=None):
    model.eval()
    predictions = []
    references = []
    input_texts = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}" if epoch else "Evaluating")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            
            encoder_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            generated_ids = model.bart.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=CONFIG['max_target_length'],
                num_beams=CONFIG['num_beams'],
                early_stopping=True,
                no_repeat_ngram_size=CONFIG['no_repeat_ngram_size'],
                length_penalty=CONFIG['length_penalty'],
                repetition_penalty=CONFIG['repetition_penalty']
            )
            
            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            batch_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            predictions.extend(batch_preds)
            references.extend(batch_refs)
            input_texts.extend(batch_inputs)
    
    metrics = calculate_metrics(predictions, references)
    
    # Log samples
    logging.info("\n=== Sample Predictions ===")
    samples = []
    for i in range(min(5, len(predictions))):
        sample = {
            "input": input_texts[i],
            "reference": references[i],
            "predicted": predictions[i]
        }
        samples.append(sample)
        logging.info(f"\nInput: {input_texts[i]}")
        logging.info(f"Reference: {references[i]}")
        logging.info(f"Predicted: {predictions[i]}")
        logging.info("-"*50)
    
    # Save results
    if epoch is not None:
        results_df = pd.DataFrame({
            'input': input_texts,
            'reference': references,
            'prediction': predictions
        })
        results_path = os.path.join(CONFIG['results_path'], f'val_results_epoch_{epoch+1}.csv')
        results_df.to_csv(results_path, index=False)
        
        metrics_path = os.path.join(CONFIG['results_path'], f'val_metrics_epoch_{epoch+1}.csv')
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        
        # Add to report
        if epoch == CONFIG['num_epochs'] - 1:  # Last epoch
            report["sample_predictions"] = samples
    
    return metrics

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs, device, tokenizer):
    best_metrics = {'bertscore_f1': 0}
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device),
                pixel_values=batch['pixel_values'].to(device)
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(CONFIG['model_save_path'], f'model_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        
        # Validation
        val_metrics = evaluate_model(model, val_dataloader, device, tokenizer, epoch)
        
       
        epoch_report = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "validation_metrics": val_metrics
        }
        report["training"]["epochs"].append(epoch_report)
        
        # Save best model
        if val_metrics['bertscore_f1'] > best_metrics['bertscore_f1']:
            best_metrics = val_metrics
            best_model_path = os.path.join(CONFIG['model_save_path'], 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with BERTScore F1: {val_metrics['bertscore_f1']:.4f}")
            report["best_model"] = {
                "epoch": epoch + 1,
                "metrics": val_metrics,
                "path": best_model_path
            }
    
    return best_metrics

def save_report():
    report_path = os.path.join(CONFIG['report_path'], 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logging.info(f"Saved training report to {report_path}")


# 3.6 Testing and Model Inference
# A test.tsv file, formatted similarly to train df.tsv, along with D test.pkl and O test.pkl files, will be
# provided for evaluation. Students are required to implement a function that:
# • Loads the trained model.
# • Processes these files.
# • Generates and stores sarcasm explanations in an output file.
# • Computes and reports all evaluation metrics.

def run_inference(model_path, test_df_path, D_test_path, O_test_path, output_path):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    model = TURBOMuSE()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    test_df = pd.read_csv(test_df_path, sep='\t')
    with open(D_test_path, 'rb') as f:
        D_test = pickle.load(f)
    with open(O_test_path, 'rb') as f:
        O_test = pickle.load(f)
    
    test_dataset = MuSEDataset(test_df, D_test, O_test, tokenizer, feature_extractor, CONFIG['image_dir'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            
            encoder_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            generated_ids = model.bart.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=CONFIG['max_target_length'],
                num_beams=CONFIG['num_beams'],
                early_stopping=True,
                no_repeat_ngram_size=CONFIG['no_repeat_ngram_size'],
                length_penalty=CONFIG['length_penalty'],
                repetition_penalty=CONFIG['repetition_penalty']
            )
            
            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)
            
            if 'labels' in batch:
                batch_refs = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                references.extend(batch_refs)
    
    test_df['predicted_explanation'] = predictions
    test_df.to_csv(output_path, sep='\t', index=False)
    
    if len(references) > 0:
        metrics = calculate_metrics(predictions, references)
        logging.info("\nTest Set Metrics:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        metrics_path = os.path.join(CONFIG['results_path'], 'test_metrics.csv')
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    
    return test_df

def main():
    logging.info("Starting MuSE model training pipeline")
    
    # Loadign yhe data
    train_df = pd.read_csv('train_df.tsv', sep='\t')
    val_df = pd.read_csv('val_df.tsv', sep='\t')
    
    with open('D_train.pkl', 'rb') as f:
        D_train = pickle.load(f)
    with open('O_train.pkl', 'rb') as f:
        O_train = pickle.load(f)
    with open('D_val.pkl', 'rb') as f:
        D_val = pickle.load(f)
    with open('O_val.pkl', 'rb') as f:
        O_val = pickle.load(f)
    
    # Initializng the tokenizer and feature extractor adb data creation
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    train_dataset = MuSEDataset(train_df, D_train, O_train, tokenizer, feature_extractor, CONFIG['image_dir'])
    val_dataset = MuSEDataset(val_df, D_val, O_val, tokenizer, feature_extractor, CONFIG['image_dir'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    
    # Initialize model
    model = TURBOMuSE().to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=len(train_loader) * CONFIG['num_epochs']
    )
    
    # Train model
    logging.info("Starting training...")
    best_metrics = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        CONFIG['num_epochs'],
        device,
        tokenizer
    )
    
    logging.info(f"Training complete. Best validation metrics: {best_metrics}")
    save_report()


# To simplify implementation, a preprocessed dataset will be provided, including:
# • All images in the dataset
# • Train, and validation pickle files containing:
# – Image descriptions (e.g. : D train.pkl)
# – Detected objects (e.g. : O train.pkl)
# • Data files (e.g. : train df.tsv) with fields: post ID (pid), text, explanation, and sarcasm target
# Access the dataset here: MORE+ Dataset

if __name__ == '__main__':
    print("MuSE model training and inference pipeline")
    mode = int(input("Enter 1 for training and 2 for inference: "))
    
    if mode == 1:
        main()
    else:
        print("Running inference with default paths...")
        run_inference(
            model_path=os.path.join(CONFIG['model_save_path'], 'best_model.pt'),
            test_df_path='val_df.tsv',
            D_test_path='D_val.pkl',
            O_test_path='O_val.pkl',
            output_path=os.path.join(CONFIG['results_path'], 'test_predictions.tsv')
        )