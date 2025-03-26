import torch
import torch.nn as nn
import logging
from tqdm import tqdm

from src import config

class __TrainingResults:
    """
    Stores training history, including loss, acc, lr, and epoch info
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lr_history = []
        self.epochs = []

    def add_results(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        """
        Adds the results of a single epoch to the stored history
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.lr_history.append(lr)
            

def train_model(model, train_loader, valid_loader, lr=0.001, num_epochs=10, device=config.DEVICE, model_path=config.MODEL_PATH):
    """
    Trains the model and records training history
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for the training dataset
        valid_loader: DataLoader for the validation dataset
        lr (float, optional): Learning rate. Defaults to 0.001.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        device: Device to use for training (e.g., 'cuda' or 'cpu').
    
    Returns:
        tuple: Trained model and TrainingResults instance containing training history
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    results = __TrainingResults()
    
    model = model.to(device)
    
    # Track the best validation loss (minimized)
    valid_loss_min = float('inf') # set initial minimum to infinity

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        correct_train = 0
        total_train = 0
        correct_valid = 0
        total_valid = 0
        
        # Training phase
        model.train() # turn on dropout for training
        
        # Wrap train_loader with tqdm for progress bar
        prog_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=False)
        
        for images, labels in prog_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
                        
            _, predicted = torch.max(output, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
            prog_bar.set_postfix(loss=loss.item())
        
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = (correct_train / total_train) * 100        
        cur_lr = optimizer.param_groups[0]['lr']
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(output, 1)
            correct_valid += (predicted == labels).sum().item()
            total_valid += labels.size(0)
        
        val_loss /= len(valid_loader)
        val_acc = (correct_valid / total_valid) * 100
        
        # Adjust learning rate scheduler if necessary
        if hasattr(model, 'scheduler'):
            model.scheduler.step(val_loss)
        
        # Log training/validation info
        logging.info(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}%"
            + f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}%"
            + f" | Lr: {cur_lr}"
        )
        
        # Add results to history
        results.add_results(
            epoch=epoch+1, 
            train_loss=epoch_loss, 
            train_acc=epoch_acc, 
            val_loss=val_loss, 
            val_acc=val_acc,
            lr=cur_lr
        )
        
        # Save model if validation loss decreased
        if val_loss <= valid_loss_min:
            logging.info(f"Validation loss decreased from {valid_loss_min:.4f} to {val_loss:.4f} - Saving model")
            model_name = type(model).__name__
            torch.save(model.state_dict(), model_path / f"{model_name}_model.pth")
            valid_loss_min = val_loss
    
    return model, results
