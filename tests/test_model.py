import torch
import torchvision
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model import Net  # Now Python can find the model module


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_model_parameters():
    model = Net()
    param_count = count_parameters(model)
    assert (
        param_count < 25000
    ), f"Model has {param_count} parameters, should be less than 25000"


def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your pre-trained model
    model = Net()
    try:
        model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
    except:
        raise Exception(
            "Could not find model.pth. Please ensure the model is saved in the root directory."
        )

    model.to(device)
    model.eval()

    # Load test dataset
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        batch_size=1000,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    assert accuracy > 99.4, f"Model accuracy is {accuracy}%, should be > 99.4%"
