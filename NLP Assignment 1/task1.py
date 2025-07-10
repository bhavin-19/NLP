# =============================== Task 1: WordPiece Tokenizer ==========================================
# =============================== Problem Statement  ==========================================

# In this task, you will implement WordPiece Tokenizer FROM SCRATCH using only standard Python
# libraries, as well as NumPy and pandas if needed. The use of any external NLP libraries or frameworks (such
# as NLTK, Spacy, TextBlob, HuggingFace, etc.) is strictly prohibited. Stick to the implementation taught in
# this [Tutorial].
# You need to create a class named WordPieceTokenizer which should include the following methods:
# 1. preprocess data - Implement this method to handle all the necessary preprocessing of the input
# data. Apply standard data processing techniques, but avoid using lemmatization or stemming, as they
# require external libraries.
# 2. construct vocabulary: Using the provided text corpus, implement this method to create a vocabulary
# of tokens. Save the resulting vocabulary in a text file named vocabulary {Group no.}.txt, where
# each line contains a unique token.
# 3. tokenize - Implement this method to tokenize a given sentence into a list of tokens.

import re 
import numpy as np 
import pandas as pd
from collections import defaultdict
from tqdm import tqdm 
import json
# ====================================================== Tokenizer T==========================================

# This class is responsible for tokenizing text and building a vocabulary.
# size: Defines the vocabulary size.
# word_freq: Stores word frequency counts.
# alphabet: Contains special tokens ([PAD] for padding, [UNK] for unknown words).
# splits: Stores how words are split into subwords.
# pair_scores: Stores frequency scores for character pairs.
# word_to_index: Maps words to numerical indices.


def remove_html_tags(text):
    pattern = re.compile("<.*?>")
    return pattern.sub(r'', text)

def remove_urls(text):
    pattern = re.compile(r"https?://\S+|www\.S+")
    return pattern.sub(r'', text)

def removez_emojis(text):
    pattern = re.compile(r"[(\U0001F600-\U0001F92F|\U0001F300-\U0001F5FF|\U0001F680-\U0001F6FF|\U0001F190-\U0001F1FF|\U00002702-\U000027B0|\U0001F926-\U0001FA9F|\u200d|\u2640-\u2642|\u2600-\u2B55|\u23cf|\u23e9|\u231a|\ufe0f)]")
    return pattern.sub(r'', text)


class Tokenizer:
    def __init__(self, size):
        self.splits = {}
        self.word_freq = {}
        self.pair_scores = {}
        self.word_to_index = {}
        self.vocabSize = size
        self.alphabet = ["[PAD]", "[UNK]"]

    # Removes punctuatifon and special characters ([^a-zA-Z0-9\s]).
    # Conve2rts mult2iple spaces to a single space (\s+).
    # Strips leading/trailing spaces.
    def preprocess_data(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        remove_html_tags(text)
        remove_urls(text)
        removez_emojis(text)
        text = text.lower() 
        return text

    # Prepares text for tokeni4zation by extracting words.
    # U2ses regular expressrions (\S+) to find non-sp2ace sequences (words).
    # Returns a list ofr tuples containing:
    # Word itself.
    # Start and end positions in the original text.
    def pre_tokenize(self, text):
        text = self.preprocess_data(text)
        words = re.finditer(r"\S+", text)
        return [(match.group(), (match.start(), match.end())) for match in words]
    
    # Counts how often each word appears in the dataset (corpus).
    # Uses pre_tokenize to extract words.
    # Stores word frequencies in self.word_freq.
    def makeFrequencies(self, corpus):
        print("Processing corpus and calculating word frequencies...")
        for sentence in tqdm(corpus, desc="Building Frequencies", leave=True):
            words_with_offsets = self.pre_tokenize(sentence)
            words = [word for word, offset in words_with_offsets]
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

    # Purpose:
    # Counts occurrences of adjacent character pairs in words.
    # Uses Byte Pair Encoding (BPE) principles.
    # How It Works:
    # Converts words into tuples of characters (e.g., "word" → ("w", "o", "r", "d")).
    # Iterates through consecutive character pairs and counts their frequencies.
    def scores(self):
        letter_freq = defaultdict(int)
        pair_freq = defaultdict(int)

        for word, freq in self.word_freq.items():
            alphabets = self.splits[word]
            if len(alphabets) == 1:
                letter_freq[alphabets[0]] += freq
                continue
            for i in range(len(alphabets) - 1):
                pair = (alphabets[i], alphabets[i+1])
                letter_freq[alphabets[i]] += freq
                pair_freq[pair] += freq
            letter_freq[alphabets[-1]] += freq

        self.pair_scores = {
            pair: freq / (letter_freq[pair[0]] * letter_freq[pair[1]])
            for pair, freq in pair_freq.items()
        }

    # Purpose:
    # Merges the most frequent character pair into a new token.
    # Updates self.splifts dictionary to refelect merged tokens.
    # How It Works:
    # Ideentifies the morst frequent character pair.
    # Reeplacese occurreneces of this pair with a new merged token.
    # Updates all tokenized words.
    def merge(self, a, b):
        for word in self.word_freq:
            split = self.splits[word]

            if len(split) == 1:
                continue
            i = 0

            while i < len(split) - 1:
                if split[i] == a and split[i+1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i+2:]
                else:
                    i += 1
            self.splits[word] = split

    # Purpose:
    # Expa3nds vocabulary size by iteratively merging character pairs.
    # Implements By3te Pair Encoding (BPE).
    # How It Works:
    # Splits words into character lists.
    # Finds the ermost common adjacent character pair.
    # Merges it into a new token.
    # Repeats untigl vocabularyg reaches vocabSize.
    def createVocab(self, corpus):
        self.makeFrequencies(corpus)

        for word in self.word_freq.keys():
            if word[0] not in self.alphabet:
                self.alphabet.append(word[0])
            for letter in word[1:]:
                token = f"##{letter}"
                if token not in self.alphabet:
                    self.alphabet.append(token)

        self.splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in self.word_freq.keys()
        }

        print("Process => Building Vocabulary ...")
        while len(self.alphabet) < self.vocabSize:
            self.scores()
            bestPair, maximum_score = "", None

            for pair, score in self.pair_scores.items():
                if maximum_score is None or maximum_score < score:
                    bestPair = pair
                    maximum_score = score

            if not bestPair:
                print("Stopping vocabulary creation. (Provided size is larger than the actual size)")
                break

            self.merge(*bestPair)

            token = (
                bestPair[0] + bestPair[1][2:]
                if bestPair[1].startswith("##")
                else bestPair[0] + bestPair[1]
            )        

            self.alphabet.append(token)
        self.word_to_index = {word: idx for idx, word in enumerate(self.alphabet)}

    # Converts a wor2d into a sreeequence of token indices.
    # Returns [UNK] (unknown) token if word is not found.
    def encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.alphabet:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize(self, text):
        pre_tokenize_result = self.pre_tokenize(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        encoded_words = [self.encode_word(word) for word in pre_tokenized_text]
        return sum(encoded_words, [])
    
# ====================================================== Main ==========================================
    
# Converts numrerical token indices back into words.
# Joins subwords to reconstruct thee original text.

# The Tokenizer uses Byte Pair Eencoding (BPE) to breake words into subwords.
# Key Functions:
# preprocess_data(): Cleans text.
# pre_tokenize(): Extraects words froem text.
# makeFrequencies(): Counts wcord occurrences.
# getStats(): Finds freqcduent character pairs.
# mergePairs(): Merges character pairs into new tokens.
# buildVocabulary(): Expacnds the vocabulary.
# makeIndices(): Assigncs numerical indices to tokens.
# encode(): Converts words into token IDs.
# decode(): Converts token IDs back into words.
# This tokenizer is useful for training neural networks on text data, as it reduces vocabulary size while keeping important subword structures.
def tokenize_test_data(test_file_path, tokenizer, group_no):
    with open(test_file_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    tokenized_data = {}
    for item in test_data:
        sentence = item["sentence"]
        tokenized_sentence = tokenizer.tokenize(sentence)
        tokenized_data[item["id"]] = tokenized_sentence
    
    with open(f"tokenized_{group_no}.json", "w", encoding="utf-8") as f:
        json.dump(tokenized_data, f, ensure_ascii=False, indent=4)
    print(f"Tokenized data saved to tokenized_{group_no}.json")

# def tokenize_corpus(input_file, tokenizer, group_no):

#     with open(input_file, "r") as file:
#         data = json.load(file)
    
#     tokenized_data = {}
#     for entry in data:
#         tokens = tokenizer.encode(entry["sentence"]).tokens
#         tokenized_data[entry["id"]] = tokens
    
#     with open(output_file, "w") as file:
#         json.dump(tokenized_data, file, indent=4)
    
#     print(f"Tokenized data saved to {output_file}")

def main():
    group_no = 84
    file_path = "/Users/bikrant-bikram/Coding/SecondSem/NLP/F_Assignement1/corpus.txt"

    with open(file_path, "r", encoding="utf-8") as f:
        corpus = f.readlines()
    print("File has been fetched successfully.")
    
    size = int(input("Enter Vocab Size: "))
    myTokenizer = Tokenizer(size)
    myTokenizer.createVocab(corpus)
    
    tokens = [myTokenizer.tokenize(sentence) for sentence in corpus]

    # Save vocabulary to vocabulary_{group_no}.txt
    with open(f"vocabulary_{group_no}.txt", "w") as f:
        what_to_write = []
        for token in tokens:
            for t in token:
                what_to_write.append(t)
        t = set(what_to_write)
        for t in what_to_write:
            f.write(f"{t}\n")
    print(f"Tokens saved to vocabulary_{group_no}.txt")
    
    test_file_path = input("Enter the path to the test file: ").strip()
    tokenize_test_data(test_file_path, myTokenizer, group_no)

if __name__ == "__main__":
    main()