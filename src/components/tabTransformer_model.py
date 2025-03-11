import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
import time
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Configure logging
def setup_logging(log_file='tabtransformer_nids.log'):
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('tabtransformer_nids')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join('logs', log_file))
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Custom dataset class for network flow data
class NetworkFlowDataset(Dataset):
    def __init__(self, features, labels):
        # Convert pandas DataFrames to numpy arrays first
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(labels, pd.DataFrame):
            labels = labels.values.ravel()  # Flatten the labels array
            
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Multi-head Self-attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        # Reshape for multi-head attention
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(x.device)
        attention = torch.softmax(energy, dim=-1)
        
        # Apply attention to values
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        
        return self.fc(x)

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(input_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        attention_output = self.attention(x)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Feature Embedding Layer for TabTransformer
class FeatureEmbedding(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(FeatureEmbedding, self).__init__()
        self.embedding_layers = nn.ModuleList([nn.Linear(1, embedding_dim) for _ in range(num_features)])
        
    def forward(self, x):
        # Split features and embed each separately
        embeddings = []
        for i, layer in enumerate(self.embedding_layers):
            feature = x[:, i].unsqueeze(1)  # Extract single feature and maintain dimension
            embedded_feature = layer(feature)
            embeddings.append(embedded_feature)
        
        return torch.stack(embeddings, dim=1)  # [batch_size, num_features, embedding_dim]

# TabTransformer Model
class TabTransformer(nn.Module):
    def __init__(self, num_features, num_classes, embedding_dim=32, num_heads=4, 
                 num_layers=3, ff_dim=64, dropout=0.1, logger=None):
        super(TabTransformer, self).__init__()
        
        self.logger = logger
        if self.logger:
            self.logger.info(f"Initializing TabTransformer with {num_features} features, {embedding_dim} embedding dim, "
                            f"{num_heads} attention heads, {num_layers} transformer layers")
        
        self.feature_embedding = FeatureEmbedding(num_features, embedding_dim)
        
        # Stack of transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features * embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_classes)
        )
        
    def forward(self, x):
        # Embed each feature
        x = self.feature_embedding(x)
        
        # Apply transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
        
        # Flatten and classify
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.classifier(x)
        
        return x

# Function to preprocess and prepare data
def prepare_data(data_path, batch_size=64, logger=None):
    start_time = time.time()
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    
    # Compute Class Weights
    if logger:
        logger.info("Computing class weights for imbalanced data handling")
    
    classes = np.unique(y_train.values.ravel())
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y_train.values.ravel()
    )
    
    if logger:
        logger.info(f"Class weights: {class_weights}")
        
    # Create datasets
    train_dataset = NetworkFlowDataset(X_train, y_train)
    test_dataset = NetworkFlowDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    elapsed_time = time.time() - start_time
    if logger:
        logger.info(f"Data preparation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Created {len(train_loader)} training batches and {len(test_loader)} testing batches")
    
    return train_loader, test_loader, X_train.shape[1], class_weights

# Training function
def train_model(model, train_loader, test_loader, class_weights=None, epochs=10, lr=0.001, logger=None):
    start_time = time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if logger:
        logger.info(f"Using device: {device}")
        
        # Log model architecture
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model summary: Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    model = model.to(device)
    
    # Use Class Weights
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        if logger:
            logger.info(f"Using weighted CrossEntropyLoss with class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
        if logger:
            logger.info(f"Using standard CrossEntropyLoss (no class weights)")
            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if logger:
        logger.info(f"Optimizer: Adam with learning rate {lr}")
        logger.info(f"Loss function: CrossEntropyLoss")
    
    # Track metrics
    best_val_accuracy = 0
    training_history = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        batch_times = []
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            batch_start = time.time()
            
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            # Log every 50 batches
            if batch_idx % 50 == 0 and logger:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                            f"Loss: {loss.item():.4f}, "
                            f"Accuracy: {100 * (predicted == labels).sum().item() / labels.size(0):.2f}%, "
                            f"Batch time: {batch_times[-1]:.4f}s")
        
        train_accuracy = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        avg_batch_time = sum(batch_times) / len(batch_times)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        # Confusion matrix data
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        val_start = time.time()
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
                
                # Accumulate confusion matrix data
                true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
                false_positives += ((predicted == 1) & (labels == 0)).sum().item()
                false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(test_loader)
        val_time = time.time() - val_start
        
        # Calculate additional metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Save metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        training_history.append(epoch_metrics)
        
        epoch_time = time.time() - epoch_start
        
        # Log epoch results
        if logger:
            logger.info(f"Epoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f}s (train: {epoch_time-val_time:.2f}s, val: {val_time:.2f}s)")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            logger.info(f"Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1 Score: {f1_score:.4f}")
            logger.info(f"Confusion Matrix: TP={true_positives}, TN={true_negatives}, FP={false_positives}, FN={false_negatives}")
            logger.info(f"Average batch processing time: {avg_batch_time:.4f}s")
            logger.info("-" * 80)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_tabtransformer_model.pth")
            if logger:
                logger.info(f"Saved new best model with validation accuracy: {val_accuracy:.2f}%")
    
    training_time = time.time() - start_time
    if logger:
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        
        # Save training history
        history_df = pd.DataFrame(training_history)
        history_df.to_csv("training_history.csv", index=False)
        logger.info("Training history saved to training_history.csv")
    
    return model, training_history

# Main execution function
def main(data_path, batch_size=64, embedding_dim=32, num_heads=4, 
         num_layers=3, ff_dim=64, epochs=10, lr=0.001):
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting TabTransformer NIDS training")
    logger.info(f"Configuration: batch_size={batch_size}, embedding_dim={embedding_dim}, "
                f"num_heads={num_heads}, num_layers={num_layers}, ff_dim={ff_dim}, "
                f"epochs={epochs}, learning_rate={lr}")
    
    try:
        # Log system info
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Prepare data
        logger.info("Preparing data...")
        train_loader, test_loader, num_features, class_weights = prepare_data(data_path, batch_size, logger)
        
        # Initialize model
        logger.info("Initializing model...")
        model = TabTransformer(
            num_features=num_features,
            num_classes=5,  # Binary classification 
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            logger=logger
        )
        
        # Log model architecture summary
        logger.info("Model architecture initialized")
        
        # Train model
        logger.info("Starting model training...")
        model, history = train_model(model, train_loader, test_loader, class_weights, epochs, lr, logger)
        # Save final model
        torch.save(model.state_dict(), "final_tabtransformer_nids_model.pth")
        logger.info("Final model saved to final_tabtransformer_nids_model.pth")
        
        # Plot training history
        try:
            import matplotlib.pyplot as plt
            
            # Create plots directory
            os.makedirs('plots', exist_ok=True)
            
            # Plot training and validation loss
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot([x['epoch'] for x in history], [x['train_loss'] for x in history], label='Train Loss')
            plt.plot([x['epoch'] for x in history], [x['val_loss'] for x in history], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot([x['epoch'] for x in history], [x['train_accuracy'] for x in history], label='Train Accuracy')
            plt.plot([x['epoch'] for x in history], [x['val_accuracy'] for x in history], label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            plt.plot([x['epoch'] for x in history], [x['precision'] for x in history], label='Precision')
            plt.plot([x['epoch'] for x in history], [x['recall'] for x in history], label='Recall')
            plt.plot([x['epoch'] for x in history], [x['f1_score'] for x in history], label='F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Precision, Recall and F1 Score')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('plots/training_metrics.png')
            logger.info("Training metrics plot saved to plots/training_metrics.png")
            
            # Plot confusion matrix from last epoch
            plt.figure(figsize=(8, 6))
            cm = np.array([
                [history[-1]['true_negatives'], history[-1]['false_positives']],
                [history[-1]['false_negatives'], history[-1]['true_positives']]
            ])
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            classes = ['Benign', 'Attack']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig('plots/confusion_matrix.png')
            logger.info("Confusion matrix plot saved to plots/confusion_matrix.png")
            
        except Exception as e:
            logger.warning(f"Failed to create plots: {str(e)}")
        
        logger.info("Training process completed successfully")
        
        return model, history
    
    except Exception as e:
        logger.error(f"Error during training process: {str(e)}", exc_info=True)
        raise

# Example usage
if __name__ == "__main__":
    # Replace with your dataset path
    data_path = "data/processed/dataset.csv"
    
    # Train the model
    model, history = main(
        data_path=data_path,
        batch_size=64,
        embedding_dim=32,
        num_heads=4,
        num_layers=3,
        ff_dim=64,
        epochs=10
    )