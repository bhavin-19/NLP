import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

# Set Device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.readlines()

# Initialize WordPiece Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenize Corpus
tokenized_corpus = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")

# Create Vocabulary Mapping
word_to_idx = tokenizer.get_vocab()
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(word_to_idx)

# Define CBOW Model
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, context_words):
        embedded = self.embeddings(context_words)
        context_vector = embedded.mean(dim=1)
        output = self.linear(context_vector)
        return self.softmax(output)

# Word2Vec Dataset Class
class Word2VecDataset:
    def __init__(self, tokenized_corpus, window_size=2):
        self.window_size = window_size
        self.cbow_data = []

        for sentence in tokenized_corpus["input_ids"]:
            sentence = sentence.tolist()
            for i, target_word in enumerate(sentence):
                context_words = sentence[max(0, i - window_size):i] + sentence[i+1:i + window_size+1]
                if len(context_words) == window_size * 2:
                    self.cbow_data.append((context_words, target_word))

    def __len__(self):
        return len(self.cbow_data)

    def __getitem__(self, index):
        context_words, target_word = self.cbow_data[index]
        return torch.tensor(context_words, dtype=torch.long), torch.tensor(target_word, dtype=torch.long)

# Create Dataset and DataLoader
dataset = Word2VecDataset(tokenized_corpus, window_size=2)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# Initialize Model and Move to GPU
embedding_dim = 300
model = CBOWModel(vocab_size, embedding_dim).to(device)

# Training Function
def train_cbow(model, dataloader, num_epochs=10, learning_rate=0.001):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for context, target in dataloader:
            context, target = context.to(device), target.to(device)  # Move tensors to GPU
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# Train Model on GPU
train_cbow(model, dataloader, num_epochs=25)

# Extract and Display Word Embeddings
embeddings = model.embeddings.weight.data.cpu()  # Move embeddings to CPU before printing

print("\nWord Embeddings (Example):")
for word, idx in list(word_to_idx.items())[:10]:
    print(f"Word: {word}, Embedding: {embeddings[idx]}")
