import argparse

from matplotlib import pyplot as plt

from src import config
from src.dataset import MnistDataset, MnistEDA
from src.modeling.models import CNN, FFNN, initialize_weights_normal
from src.modeling.train import train_model, test_model, plot_training_history

def get_args():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10 dataset")
    parser.add_argument(
        "--model",
        choices=["cnn", "ffnn"],
        required=True,
        help="Specify which model to train ('cnn' or 'ffnn')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train the model (default: 30)",
    )
    parser.add_argument(
        "--debug_mode",
        action='store_true',
        help="Enable debug mode."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model = args.model
    epochs = args.epochs
    debug_mode=args.debug_mode
    
    # Get the datasets
    dataset = MnistDataset()
    train_data, test_data, class_names = dataset.get_datasets()
    train_loader, valid_loader, test_loader = dataset.data_loader(
        train_data, test_data, valid_size=0.2
    )
    
    if debug_mode:
        eda = MnistEDA()

        # Plot class distribution and pie chart for the training data
        fig = eda.plot_class_distribution_and_pie_chart(train_data, class_names)
        plt.show()

        # Display 4 number of images their labels
        fig = eda.plot_images(train_data, class_names, num_imgs=4)
        plt.show()

        # Display images and labels of a batch
        fig = eda.plot_batch_images(train_loader)
        plt.show()
    
    if model == "cnn":
        model = CNN()
        model.apply(initialize_weights_normal)
        model_folder = config.RESULTS_PATH / "cnn"
        model_folder.mkdir(parents=True, exist_ok=True) 
    elif model == "ffnn":
        model = FFNN()
        model.apply(initialize_weights_normal)
        model_folder = config.RESULTS_PATH / "ffnn"
        model_folder.mkdir(parents=True, exist_ok=True)
    
    model, results = train_model(
        model,
        train_loader,
        valid_loader,
        lr=0.001,
        num_epochs=epochs,
        device=config.DEVICE
    )
    
    fig = plot_training_history(results)
    fig.savefig(model_folder / "training_history.png")
    plt.close(fig)
    
    fig = test_model(model, test_loader)
    fig.savefig(model_folder / "test_results.png")
    plt.close(fig)
