import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class CommentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=128):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',  # Use fixed-length padding for consistency
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Remove the extra batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_model(model, train_loader, val_loader, device, epochs=5):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        predictions_all = []
        labels_all = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions_all.extend(preds.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = accuracy_score(labels_all, predictions_all)
        f1 = f1_score(labels_all, predictions_all, average='weighted')

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.4f}')
        print(f'Validation F1 Score: {f1:.4f}')
        print('-------------------')

    return model


def main():
    # Load the dataset (update the path if needed)
    df = pd.read_csv('comments_dataset_for_training.csv')

    # Expected CSV columns: 'comment_text' and 'label'
    # If the CSV has string labels (e.g., "Troll", "Fan"), map them accordingly:
    if df['label'].dtype == object:
        # Mapping: Troll -> 0, Fan -> 1
        label_mapping = {'Troll': 0, 'Fan': 1}
        df['label'] = df['label'].map(label_mapping)
    else:
        # If labels are already numeric, ensure they are integers.
        df['label'] = df['label'].astype(int)

    # Drop any rows with missing data
    df = df.dropna(subset=['comment_text', 'label'])

    # Use stratification if more than one class is present to maintain class distribution.
    stratify_col = df['label'] if df['label'].nunique() > 1 else None

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['comment_text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=stratify_col
    )

    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Create the datasets and dataloaders
    train_dataset = CommentDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = CommentDataset(val_texts, val_labels, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model
    model = train_model(model, train_loader, val_loader, device, epochs=5)

    # Save the trained model and tokenizer
    model.save_pretrained('trained_bert_model')
    tokenizer.save_pretrained('trained_bert_model')

    print("Training completed and model saved!")


if __name__ == "__main__":
    main()
