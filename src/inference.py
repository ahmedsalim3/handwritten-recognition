import argparse
import logging
import torch

from src.modeling.models import FFNN, CNN
from src import config
from src.modeling.infer import predict_and_visualize

if __name__ == "__main__":
    device = config.DEVICE

    parser = argparse.ArgumentParser(
        description="Run image inference with CNN or FFNN model"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "ffnn"],
        required=True,
        help="Choose the model: cnn or ffnn",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image (URL or local file)",
    )
    parser.add_argument(
        "--requests",
        action="store_true",
        help="If the image is a URL, use requests to fetch it",
    )

    args = parser.parse_args()

    if args.model == "cnn":
        model = CNN()
        model_path = config.MODEL_PATH / "CNN_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    else:
        model = FFNN()
        model_path = config.MODEL_PATH / "FFNN_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    logging.info("Model loaded successfully.")

    result = predict_and_visualize(
        model=model,
        image_path=args.image_path,
        request=args.requests,
    )
    logging.info(result)
