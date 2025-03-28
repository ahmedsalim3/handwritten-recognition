import logging
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from src import config

def predict_and_visualize(model, image_path, top_N=1, request=False):
    """
    Function to handle inference on an image and visualize the result with top-N predictions.
    It can handle both local image paths or download an image from the web if `request=True`.

    Args:
        model: The pre-trained model to make predictions (either FFNN or CNN).
        image_path: The path to the image file (local path or URL).
        top_N (int): The top N predictions to display.
        request (bool): If True, the function will attempt to download the image from the web.
        device: Device to run the inference on ("cuda" or "cpu").
        
    Returns:
        Dictionary with top labels and their confidences.
    """
    
    model.eval()

    # Default transformations to match the input expected by FFNN/CNN
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for single-channel images
    ])

    # Load image
    if request:
        try:
            response = requests.get(image_path)
            response.raise_for_status()
            org_img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image from URL: {e}")
            return None
    else:
        try:
            org_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image from file: {e}")
            return None

    # Apply transformations
    img = transform(org_img).unsqueeze(0)
    img = img.to(config.DEVICE)
    model = model.to(config.DEVICE)

    try:
        # Make prediction
        with torch.no_grad():
            predictions = model(img)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)[0]

        # Get top-N predictions
        top_indices = torch.argsort(probabilities, descending=True)[:top_N]
        top_labels = [f"Class {idx.item()}" for idx in top_indices]
        top_confidences = [probabilities[idx].item() for idx in top_indices]

        # Visualize the image and predictions
        plt.figure(figsize=(10, 6))
        plt.imshow(org_img)
        plt.axis("off")
        plt.title(
            "\n".join(
                [
                    f"{label}: {conf:.2f}" for label, conf in zip(top_labels, top_confidences)
                ]
            ),
            fontsize=12,
        )

        plt.tight_layout()
        plt.show()

        return {"top_labels": top_labels, "top_confidences": top_confidences}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None
