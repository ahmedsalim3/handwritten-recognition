import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FFNN(nn.Module):
    """
    FFNN Model: Feed Forward Neural Network (Fully Connected Network)
    
    This class defines a simple feed-forward neural network with the following architecture:
    
    - Input Layer: 784 nodes (flattened 28x28 input image)
    - Hidden Layer 1: 512 nodes (fully connected to the input)
    - Hidden Layer 2: 256 nodes (fully connected to the first hidden layer)
    - Output Layer: 10 nodes (for 10-class classification problem)
    
    The model uses the ReLU activation function for hidden layers and a softmax activation 
    (implicitly) in the output layer for classification.
    
    **Dropout Regularization:**
    Dropout with a probability of 20% is applied after each hidden layer to reduce overfitting 
    during training. This means 20% of the nodes in the hidden layers will be randomly turned off 
    during each forward pass in training.

    **Training & Evaluation Modes:**
    - `train()`: During training, the model uses dropout for regularization.
    - `eval()`: For validation or testing, dropout is turned off, ensuring stable outputs.

    Args:
        None
        
    Example Usage:
        model = FFNN()
        model.train()  # Enable dropout for training
        output = model(input_data)  # Forward pass with dropout active

        model.eval()   # Disable dropout for evaluation
        output = model(input_data)  # Forward pass without dropout
    """
    def __init__(self):
        super(FFNN,self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(784, 512)  # First hidden layer (784 inputs -> 512 neurons)
        self.fc2 = nn.Linear(512, 256)  # Second hidden layer (512 inputs -> 256 neurons)
        self.out = nn.Linear(256, 10)   # Output layer (256 inputs -> 10 classes)
        
        # Dropout regularization layer (20% dropout rate)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        # Flatten input tensor
        x = x.view(-1, 28 * 28)  # Reshape the input to a 1D vector for each sample in the batch
              
        # Pass through the first hidden layer with ReLU activation and dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Pass through the second hidden layer with ReLU activation and dropout
        x = self.dropout(F.relu(self.fc2(x)))
        
        # Output layer
        x=self.out(x)
        
        return x


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for image classification.

    Architecture:
    - Conv Layer 1: Input (1 channel), Output (16 channels), Kernel (3x3), Stride (1), Padding (1) → Output: 16x28x28
    - MaxPool: Kernel (2x2), Stride (2) → Output: 16x14x14
    - Conv Layer 2: Input (16 channels), Output (32 channels), Kernel (3x3), Stride (1), Padding (1) → Output: 32x14x14
    - MaxPool: Kernel (2x2), Stride (2) → Output: 32x7x7
    - Flattened to fully connected layers:
      - FC Layer 1: Input (7x7x32), Output (512 nodes)
      - FC Layer 2: Input (512), Output (256 nodes)
      - Output Layer: 256 -> 10 (for 10-class classification)

    Dropout (20%) is applied after each fully connected layer to reduce overfitting.
    """
    
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 10)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Apply convolutional layers + pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output from convolutional layers
        x = x.view(-1, 7 * 7 * 32)

        # Apply fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        
        # Output layer
        x = self.out(x)
        
        return x


def initialize_weights_normal(m):
    """
    Initializes the weights of the network using a normal distribution,
    with mean 0 and standard deviation based on the number of input features.
    Bias is initialized to 0.
    
    Args:
        m (nn.Module): The model layer whose weights are to be initialized.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = (1.0 / np.sqrt(n))  # Standard deviation for normal distribution
        m.weight.data.normal_(0, y)  # Initialize weights
        m.bias.data.fill_(0)  # Initialize biases to 0
