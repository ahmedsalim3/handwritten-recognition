from matplotlib import pyplot as plt
import numpy as np
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


def test_model(model, test_loader, device=config.DEVICE):
    
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    test_loss = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    model = model.to(device)

    model.eval()  # Set model to evaluation mode (turns off dropout)
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item()
        
        _, pred = torch.max(output, 1)
        correct = pred.eq(labels.data.view_as(pred)).cpu().numpy()
        
        for i in range(len(labels)):
            label = labels.data[i].item()
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss / len(test_loader)
    logging.info(f'For {type(model).__name__}:')
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Correctly predicted per class: {class_correct}, Total correctly predicted: {sum(class_correct)}")
    logging.info(f"Total Predictions per class: {class_total}, Total predictions to be made: {sum(class_total)}\n")

    for i in range(10):
        if class_total[i] > 0:
            logging.info(f"Test Accuracy of class {i}: {100. * class_correct[i] / class_total[i]:.2f}% "
                  f"({int(class_correct[i])} of {int(class_total[i])} correct)")
        else:
            logging.info(f"Test Accuracy of class {i}: N/A (no test samples)")

    overall_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    logging.info(f"Overall Test Accuracy: {overall_accuracy:.2f}% "
          f"({int(np.sum(class_correct))} of {int(np.sum(class_total))} correct)")

    # Obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Get model predictions
    images, labels = images.to(device), labels.to(device)
    
    output = model(images)
    _, preds = torch.max(output, 1)

    # Convert images to NumPy
    images = images.cpu().numpy()

    # Plot the images with predictions
    fig = plt.figure(figsize=(25, 4))
    num_images = min(20, len(images))
    for idx in range(num_images):
        ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
        ax.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title("{} ({})".format(preds[idx].item(), labels[idx].item()),
                     color=("blue" if preds[idx] == labels[idx] else "red"))

    plt.tight_layout()
    return fig


def plot_training_history(results):
    """
    Plot the training and validation loss and accuracy over multiple epochs

    Args:
        results: An instance of TrainingResults containing recorded metrics
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # losses
    ax1.plot(results.epochs, results.train_losses, label="Training Loss")
    if results.val_losses:
        ax1.plot(results.epochs, results.val_losses, label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # accuracies
    ax2.plot(results.epochs, results.train_accs, label="Training Accuracy")
    if results.val_accs:
        ax2.plot(results.epochs, results.val_accs, label="Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig
