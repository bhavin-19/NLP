
# ======================================================== Problem Statement ========================================================

# Task 2 - Implement Word2vec 25 Marks
# You are tasked with building a pipeline for training a Word2Vec model using the CBOW (Continuous Bag
# of Words) approach FROM SCRATCH in PyTorch. It consist of the following components:
# 1. You are required to create a Python class named Word2VecDataset that will serve as a custom dataset
# for training the Word2Vec model. The implementaion should include the following components:

# • The custom implementation should work with PyTorch’s DataLoader to efficiently load the train-
# ing data.. You can refer this guide [Tutorial] on creating custom dataset classes in PyTroch.

# • preprocess data - In this method, you will be preprocessing the provided corpus and prepare
# the CBOW training data for training the Word2Vec model.
# • During preprocessing, you must use the WordPieceTokenizer implemented in Task 1 to tokenize
# the input text corpus.
# 2. You required to create a Python class named Word2VecModel which implement Word2Vec CBOW
# architecture from scratch using PyTorch. After training the the model, save the trained model’s
# checkpoint for later use.
# 3. Develop a function named train to manage the entire training process of the Word2Vec model. This
# function should include all the training logic.


# ======================================================== Solution ========================================================



# ======================================================== Imports ========================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from task1 import Tokenizer
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F  # Add this import





# ======================================================== Code ========================================================

# =============================================================[CBOWModel]===================================================================
# This is a neural network model for the CBOW (Continuous Bag-of-Words) approach.
# nn.Embedding(vocab_size, embedding_dim): Converts words into dense numerical vectors.
# nn.Linear(embedding_dim, vocab_size): Connects embeddings to the vocabulary.
# nn.LogSoftmax(dim=-1): Converts output scores into log probabilities.
# Forward Function:
# Converts words to embeddings.
# Averages the context word embeddings.
# Passes the averaged embedding through a linear layer.
# Uses LogSoftmax to get probability distribution over vocabulary.


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, context_words):
        embedded = self.embeddings(context_words)  # (batch_size, context_size, embedding_dim)
        context_vector = embedded.mean(dim=1)  # Average embeddings
        output = self.linear(context_vector)  # (batch_size, vocab_size)
        return self.softmax(output)


# ============================================================[Word2VecDataset]===================================================================
# Creates training data for CBOW.
# Extracts context words around each word in a sentence.
# Stores (context, target) pairs for training.

class Word2VecDataset(Dataset):
    def __init__(self, tokenized_corpus, word_to_index, window_size=2):
        self.window_size = window_size
        self.cbow_data = []
        
        sequenced_sentences = [[word_to_index[word] for word in sentence] for sentence in tokenized_corpus]

        for sentence in sequenced_sentences:
            for i, target_word in enumerate(sentence):
                context_words = sentence[max(0, i - window_size):i] + sentence[i+1:i + window_size+1]
                if len(context_words) == window_size * 2:
                    self.cbow_data.append((context_words, target_word))
    # Returns total samples.

    def __len__(self):
        return len(self.cbow_data)
    # Converts context and target words to tensors.
    def __getitem__(self, index):
        context_words, target_word = self.cbow_data[index]
        return torch.tensor(context_words, dtype=torch.long), torch.tensor(target_word, dtype=torch.long)



# Trains the CBOW model using word pairs from the dataset.
# Steps:
# Loops over multiple epochs.
# Iterates through batches of training data.
# Computes loss and updates model weights.
# Displays progress using tqdm.

def train_cbow(model, dataloader, val_dataloader, num_epochs=10, learning_rate=0.001, device=None):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_losses = []
    validation_losses = []

    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

        for context, target in progress_bar:
            context, target = context.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        training_losses.append(avg_loss)
        print(f"Epoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # Validation
        val_loss = 0
        with torch.no_grad():
            for context, target in val_dataloader:
                context, target = context.to(device), target.to(device)
                output = model(context)
                loss = criterion(output, target)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        validation_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss}")

    # Plot Loss Curve
    plt.plot(range(1, num_epochs + 1), training_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()
    plt.savefig("plot.png")


# ============================================================[Predict]===================================================================

# Predicts the most likely next words based on context.
# Steps:
# Converts words to indices.
# Passes them through the trained CBOW model.
# Applies softmax to get probability scores.
# Returns the top k predicted words.

# CBOWModel learns word relationships using word embeddings.
# Word2VecDataset processes text into training pairs.
# train_cbow() trains the model using context-target pairs.
# predict_next_word() suggests words based on context.

def predict(model, dataloader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for context, target in dataloader:
            context, target = context.to(device), target.to(device)
            output = model(context)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(target.cpu().numpy())
    return predictions, actuals

def calculate_accuracy(predictions, actuals):
    correct = (predictions == actuals).sum().item()
    return correct / len(actuals)

def interactive_cosine_similarity(word_to_idx, embeddings):
    print("Enter a pair or triplet of words to compute cosine similarity.")
    print("Type 'EXIT' to quit the program.")
    
    while True:
        user_input = input("Enter words separated by spaces (e.g., 'love hate' or 'love affectionate joy'): ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        
        words = user_input.split()
        
        if len(words) == 1:
            word1 = words[0]
            if word1 in word_to_idx:
                vec1 = embeddings[word_to_idx[word1]]
                similarities = []
                for idx, emb in enumerate(embeddings):
                    similarity = F.cosine_similarity(vec1.unsqueeze(0), emb.unsqueeze(0)).item()
                    similarities.append((list(word_to_idx.keys())[idx], similarity))
                similarities.sort(key=lambda x: x[1], reverse=True)
                print(f"Most similar words to '{word1}':")
                for idx, (similar_word, sim) in enumerate(similarities[1:6]):
                    print(f"{similar_word}: Similarity = {sim:.4f}")
            else:
                print(f"The word '{word1}' is not in the vocabulary.")
        
        elif len(words) == 2:
            word1, word2 = words
            if word1 in word_to_idx and word2 in word_to_idx:
                vec1 = embeddings[word_to_idx[word1]]
                vec2 = embeddings[word_to_idx[word2]]
                cosine_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
                print(f"Cosine similarity between '{word1}' and '{word2}': {cosine_sim:.4f}")
            else:
                print(f"One or both of the words '{word1}' and '{word2}' are not in the vocabulary.")
        
        elif len(words) == 3:
            word1, word2, word3 = words
            if word1 in word_to_idx and word2 in word_to_idx and word3 in word_to_idx:
                vec1 = embeddings[word_to_idx[word1]]
                vec2 = embeddings[word_to_idx[word2]]
                vec3 = embeddings[word_to_idx[word3]]
                analogy_vector = vec2 - vec1 + vec3
                similarities = []
                for idx, emb in enumerate(embeddings):
                    similarity = F.cosine_similarity(analogy_vector.unsqueeze(0), emb.unsqueeze(0)).item()
                    similarities.append((list(word_to_idx.keys())[idx], similarity))
                similarities.sort(key=lambda x: x[1], reverse=True)
                print(f"Analogy: '{word1}' is to '{word2}' as '{word3}' is to:")
                for idx, (similar_word, sim) in enumerate(similarities[:5]):
                    print(f"{similar_word}: Similarity = {sim:.4f}")
            else:
                print(f"One or more of the words '{word1}', '{word2}', or '{word3}' are not in the vocabulary.")
        
        else:
            print("Please enter either a pair (word1 word2) or a triplet (word1 word2 word3).")

        print("\nYou can continue by entering a pair or triplet of words, or type 'EXIT' to quit.")

def main():
    file_path = "/Users/bikrant-bikram/Coding/SecondSem/NLP/F_Assignement1/corpus.txt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading file {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        corpus = f.readlines()
    
    print("File has been fetched successfully.")
    myTokenizer = Tokenizer(25000)
    myTokenizer.createVocab(corpus)
    
    tokens = [myTokenizer.tokenize(sentence) for sentence in corpus]
    tokenized_corpus = tokens
    print("Tokenization completed.")
    
    word_to_idx = {word: idx for idx, word in enumerate(set(word for sentence in tokenized_corpus for word in sentence))}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(word_to_idx)

    dataset = Word2VecDataset(tokenized_corpus, word_to_idx, window_size=4)
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    embedding_dim = 300
    models = {"CBOWModel": CBOWModel}
    
    hyperparams = {
        "embedding_dim": [100, 200, 300],
        "lr": [0.001, 0.0001],
        "epochs": [10, 20]
    }
    
    best_accuracy = 0
    best_model = None
    best_params = None
    
    if os.path.exists("best_cbow_model.pth") and os.path.exists("best_cbow_embeddings.pth"):
        print("Loading existing best model and embeddings...")
        with open("vocab_size.txt", "r") as f:
            vocab_size = int(f.read().strip())
        best_model = CBOWModel(vocab_size, embedding_dim).to(device)
        best_model.load_state_dict(torch.load("best_cbow_model.pth"))
        embeddings = torch.load("best_cbow_embeddings.pth")
        best_params = ("CBOWModel", embedding_dim, "N/A", "N/A")
    else:
        for model_name, model_class in models.items():
            for embedding_dim in hyperparams["embedding_dim"]:
                for lr in hyperparams["lr"]:
                    for epochs in hyperparams["epochs"]:
                        print(f"Training {model_name} with embedding_dim={embedding_dim}, lr={lr}, epochs={epochs}")
                        model = model_class(vocab_size, embedding_dim).to(device)
                        train_cbow(model, train_dataloader, val_dataloader, num_epochs=epochs, learning_rate=lr, device=device)
                        
                        embeddings = model.embeddings.weight.data.cpu()
                        predictions, actuals = predict(model, train_dataloader, device)
                        accuracy = calculate_accuracy(torch.tensor(predictions), torch.tensor(actuals))
                        print(f"Accuracy for {model_name} with embedding_dim={embedding_dim}, lr={lr}, epochs={epochs}: {accuracy}")
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model
                            best_params = (model_name, embedding_dim, lr, epochs)
        
        if best_model:
            torch.save(best_model.state_dict(), "best_cbow_model.pth")
            torch.save(best_model.embeddings.weight.data.cpu(), "best_cbow_embeddings.pth")
            print(f"Best model: {best_params} with accuracy: {best_accuracy}")
            # Save the vocabulary size
            with open("vocab_size.txt", "w") as f:
                f.write(str(vocab_size))
    
    embeddings = best_model.embeddings.weight.data.cpu()
    
    print("\nWord Embeddings (Example):")
    for word, idx in list(word_to_idx.items())[:10]:
        print(f"Word: {word}, Embedding: {embeddings[idx]}")

    interactive_cosine_similarity(word_to_idx, embeddings)

if __name__ == "__main__":
    main()