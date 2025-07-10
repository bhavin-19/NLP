
# 3 Task - Fine-tuning SpanBERT and SpanBERT-CRF 30 Marks
# 3.1 Task Description
# In this task, you are required to fine-tune SpanBERT and SpanBERT-CRF for the question-answering task
# on the SQuAD v2 dataset. The objective is to extract the answer span from the given context as per the
# question asked. An example:
# {
# ’ c on te x t ’ : ’ B e y o n c G i s e l l e Knowles−C a r te r ( born September 4 , 1 9 8 1 ) i s an
# American si n g e r , s o n gw ri t e r , r e c o r d p r oduce r and a c t r e s s . Born and r a i s e d
# i n Houston , Texas , she pe r fo rmed i n v a ri o u s s i n g i n g and d ancin g
# c om p e ti ti o n s a s a c hil d , and r o s e t o fame i n the l a t e 1990 s a s l e a d
# s i n g e r o f R&B g i r l −group De s tin y \ ’ s Child . Managed by he r f a t h e r , Mathew
# Knowles , the group became one o f the world \ ’ s be s t−s e l l i n g g i r l g r oup s o f
# a l l time . Thei r hi a t u s saw the r e l e a s e o f B e y o n c \ ’ s debut album ,
# Dange rou sly i n Love ( 2 0 0 3 ) , which e s t a b l i s h e d he r a s a s o l o a r t i s t
# worldwide , e a rned f i v e Grammy Awards and f e a t u r e d the Bill b o a r d Hot 100
# number−one s i n g l e s ”Crazy i n Love ” and ”Baby Boy ” . ’ ,
# ’ q u e s ti o n ’ : ’When did Beyonce s t a r t becoming p opul a r ? ’ ,
# ’ answers ’ : { ’ te x t ’ : [ ’ i n the l a t e 1990 s ’ ] , }
# 3.2 Dataset Description
# The dataset used for this task is the Stanford Question Answering Dataset v2 (SQuAD v2). This dataset
# consists of question-answer pairs, including unanswerable questions where no valid answer exists in the given
# passage. Due to the large size of the dataset, you may use a subset of at least 15,000 samples for training.
# 3.3 Model Training
# Both SpanBERT and SpanBERT-CRF should be fine-tuned using an appropriate subset of SQuAD v2. The
# training process must be well-documented, including hyperparameters, optimization techniques, and any
# preprocessing steps applied. Training and validation loss plots should be included in the report to illustrate
# model performance over time. Minimum epochs for training is 6.
# 3.4 Evaluation Criteria

# The models will be evaluated using the exact-match (EM) metric which measures the percentage of predic-
# tions that exactly match the ground truth. This metric should be reported on the validation set to assess

# the effectiveness of each model. Use the following code for the metric
# def exact_match_score(predictions, references):
# assert len(predictions) == len(references), "Lists must have the same length"
# matches = sum(p == r for p, r in zip(predictions, references))
# return matches / len(references) * 100 # Convert to percentage





import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, AutoModel, TrainingArguments, Trainer, 
    DataCollatorForTokenClassification, EarlyStoppingCallback
)
from datasets import load_dataset
from torchcrf import CRF
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid



HYPERPARAM_GRID = [
    {"learning_rate": 3e-5, "batch_size": 18, "epochs": 2},
    {"learning_rate": 3e-5, "batch_size": 8, "epochs": 4},
    {"learning_rate": 5e-5, "batch_size": 8, "epochs": 8},
    {"learning_rate": 3e-5, "batch_size": 16, "epochs": 16},
    {"learning_rate": 5e-5, "batch_size": 16, "epochs": 32},
]



class SpanBERTCRF(nn.Module):
    def __init__(self, model_name, num_labels=3):  # 0: outside, 1: start, 2: inside
        super(SpanBERTCRF, self).__init__()
        self.spanbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.spanbert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.spanbert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.fc(sequence_output)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction="mean")
            return {"loss": loss}
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions


def preprocess_crf(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    labels = []
    answers = examples["answers"]

    for i, idx in enumerate(sample_map):
        answer = answers[idx]
        label = [0] * len(inputs["input_ids"][i])

        if answer["text"]:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            offset = inputs["offset_mapping"][i]
            start_token = -1
            end_token = -1
            for token_idx, (s, e) in enumerate(offset):
                if s <= start_char < e:
                    start_token = token_idx
                if s < end_char <= e:
                    end_token = token_idx
                if start_token != -1 and end_token != -1:
                    break

            if start_token != -1 and end_token != -1:
                label[start_token] = 1
                for j in range(start_token + 1, end_token + 1):
                    if j < len(label):
                        label[j] = 2

        labels.append(label)

    inputs["labels"] = labels
    return inputs

def exact_match_score(predictions, references):
    assert len(predictions) == len(references), "List r of different length"
    matches = sum(p == r for p, r in zip(predictions, references))
    return matches / len(predictions) * 100 if predictions else 0

# Function to plot loss
def plot_loss(log_history, filename="loss_curve.png"):
    train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o")
    plt.plot(range(1, len(eval_losses) + 1), eval_losses, label="Validation Loss", marker="s")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()


def main1():
    print("LOading the dataset...")
    dataset = load_dataset("squad_v2")
    # The dataset used for this task is the Stanford Question Answering Dataset v2 (SQuAD v2). This dataset
    # consists of question-answer pairs, including unanswerable questions where no valid answer exists in the given
    # passage. Due to the large size of the dataset, you may use a subset of at least 15,000 samples for training.
    train_data = dataset["train"].shuffle(seed=42).select(range(15000))
    val_data = dataset["validation"].select(range(1000))
    print("Preprocessing training datdda..")
    model_name = "SpanBERT/spanbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)



    # Apply preprocessing
    train_crf = train_data.map(lambda x: preprocess_crf(x, tokenizer), batched=True, remove_columns=train_data.column_names)
    val_crf = val_data.map(lambda x: preprocess_crf(x, tokenizer), batched=True, remove_columns=val_data.column_names)

    train_crf.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_crf.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
  
    all_log_histories = []
    labels = []
    em_scores = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for config in HYPERPARAM_GRID:
        print(f"\nTraining with config: {config}")

        crf_model = SpanBERTCRF(model_name).to(device)

        training_args = TrainingArguments(
            output_dir=f"spanbert_crf_lr{config['learning_rate']}_bs{config['batch_size']}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=8,
            num_train_epochs=config["epochs"],
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)
        crf_trainer = Trainer(
            model=crf_model,
            args=training_args,
            train_dataset=train_crf,
            eval_dataset=val_crf,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        crf_trainer.train()
        crf_trainer.save_model(training_args.output_dir)

        eval_results = crf_trainer.evaluate()
        eval_loss = eval_results["eval_loss"]

        # Run inference and compute EM score
        predictions = []
        references = []

        for example in val_data:
            context = example["context"]
            question = example["question"]
            ref_answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
            
            inputs = tokenizer(question, context, return_tensors="pt", max_length=384, truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            crf_model.eval()
            with torch.no_grad():
                pred_tags = crf_model(inputs["input_ids"], inputs["attention_mask"])[0]

            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            pred_answer_tokens = [tokens[i] for i in range(len(pred_tags)) if pred_tags[i] in [1, 2]]
            pred_answer = tokenizer.convert_tokens_to_string(pred_answer_tokens)

            if ref_answer:
                predictions.append(pred_answer)
                references.append(ref_answer)

        em_score = exact_match_score(predictions, references)
        print(f"Exact Match Score: {em_score:.2f}%")

        all_log_histories.append(crf_trainer.state.log_history)
        labels.append(f"LR: {config['learning_rate']}, BS: {config['batch_size']}")
        em_scores.append(em_score)

    # Plot loss and EM scores
    plot_loss(all_log_histories[0], "crf_loss.png")

    plt.figure(figsize=(10, 5))
    plt.bar(labels, em_scores, color="skyblue")
    plt.xlabel("Hyperparameter Configurations")
    plt.ylabel("Exact Match Score (%)")
    plt.title("CRF Exact Match Score Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("crf_em_scores.png")
    plt.show()
    








# Filtyer for the Exact_match_score
def exact_match_score1(predictions, references):
    pred_answers = []
    ref_answers = []
    for pred, ref in zip(predictions, references):
        if ref == "":
            continue
        pred_answers.append(pred)
        ref_answers.append(ref) 
    return exact_match_score(pred_answers, ref_answers)

# def exact_match_score(predictions, references):
#     assert len(predictions) == len(references), "Lists must have the same length"
#     matches = sum(p == r for p, r in zip(predictions, references))
#     return matches / len(predictions) * 100

def plot_em_scores(em_scores, labels, filename="em_scores_comparison.png"):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, em_scores, color="skyblue")
    plt.xlabel("Hyperparameter Configurations")
    plt.ylabel("Exact Match Scor")
    plt.title("Exact Match Score Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(filename)
    plt.show()

# Function to preprocess the dataset
def preprocess_qa(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    start_positions = []
    end_positions = []
    answers = examples["answers"]

    for i, idx in enumerate(sample_map):
        answer = answers[idx]
        if not answer["text"]:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        if start_char < 0 or end_char > len(examples["context"][idx]):
            start_positions.append(0)
            end_positions.append(0)
            continue

        offset = inputs["offset_mapping"][i]
        start_token = -1
        end_token = -1
        for token_idx, (s, e) in enumerate(offset):
            if s <= start_char < e:
                start_token = token_idx
            if s < end_char <= e:
                end_token = token_idx
            if start_token != -1 and end_token != -1:
                break

        if start_token == -1 or end_token == -1:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(start_token)
            end_positions.append(end_token)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Function to evaluate model performance
def evaluate_qa(model, tokenizer, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []
    references = []

    for example in dataset:
        inputs = tokenizer(
            example["question"],
            example["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()

        start_idx = np.argmax(start_logits)
        end_idx = np.argmax(end_logits)

        if start_idx > end_idx:
            pred_answer = ""
        else:
            start_char = offset_mapping[start_idx][0]
            end_char = offset_mapping[end_idx][1]
            pred_answer = example["context"][start_char:end_char]

        ref_answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
        if ref_answer == "":
            continue
        else:
            predictions.append(pred_answer)
            references.append(ref_answer)

    exact_match = exact_match_score1(predictions, references)
    return exact_match

# Function to plot all models' loss curves
def plot_all_models_loss(log_histories, labels):
    plt.figure(figsize=(12, 6))

    for log_history, label in zip(log_histories, labels):
        train_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
        plt.plot(range(1, len(train_loss) + 1), train_loss, label=label, marker='o')

    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Comparison of Training Loss Across Different Hyperparameters")
    plt.legend()
    plt.grid()
    plt.savefig("hyperparam_comparison.png")
    plt.show()


def main2():
    dataset = load_dataset("squad_v2")
    # The dataset used for this task is the Stanford Question Answering Dataset v2 (SQuAD v2). This dataset
    # consists of question-answer pairs, including unanswerable questions where no valid answer exists in the given
    # passage. Due to the large size of the dataset, you may use a subset of at least 15,000 samples for training.
    train_data = dataset["train"].shuffle(seed=42).select(range(15000))
    val_data = dataset["validation"].select(range(1000))

    model_name = "SpanBERT/spanbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_qa = train_data.map(lambda x: preprocess_qa(x, tokenizer), batched=True, remove_columns=train_data.column_names)
    val_qa = val_data.map(lambda x: preprocess_qa(x, tokenizer), batched=True, remove_columns=val_data.column_names)

    all_log_histories = []
    labels = []
    em_scores = []

    for config in HYPERPARAM_GRID:
        print(f"Training with config: {config}")

        qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        training_args = TrainingArguments(
            output_dir=f"qa_model_lr{config['learning_rate']}_bs{config['batch_size']}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            num_train_epochs=config["epochs"],
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",  
            greater_is_better=False,  
        )
        # Trininh loop 
        # Usrt eatly stoppinh to ovoid overfitting
        qa_trainer = Trainer(
            model=qa_model,
            args=training_args,
            train_dataset=train_qa,
            eval_dataset=val_qa,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        qa_trainer.train()
        # Saving the models
        qa_trainer.save_model(training_args.output_dir)

        val_score = evaluate_qa(qa_model, tokenizer, val_data)
        print(f"Exact Match Score: {val_score:.2f}%")

        all_log_histories.append(qa_trainer.state.log_history)
        labels.append(f"LR: {config['learning_rate']}, BS: {config['batch_size']}")
        em_scores.append(val_score)

    plot_all_models_loss(all_log_histories, labels)
    plot_em_scores(em_scores, labels)


if __name__ == "__main__":
    main1()
    main2()