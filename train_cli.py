
import argparse
from train import train

def main():
    parser = argparse.ArgumentParser(description="Train the Probabilistic Depth Estimation Model")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone model to use (default: resnet18)')
    args = parser.parse_args()

    config = {
        "data_path": args.data_path,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "backbone": args.backbone
    }

    train(config)

if __name__ == "__main__":
    main()
