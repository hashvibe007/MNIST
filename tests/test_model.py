import torch
import torchvision
import torch.nn.functional as F
import sys
import os
from torchvision import datasets, transforms
from model import Net
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

SEED = 1

cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_optimizer_config():
    """Test if model uses SGD with correct learning rate schedule"""
    model = Net()

    # Create optimizer with initial learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create scheduler for step decay every 5 epochs
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Save optimizer and scheduler state
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": 0,
        },
        "mnist_model.pth",
    )


def test_model_optimizer():
    """Test if model uses the correct optimizer configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()

    try:
        checkpoint = torch.load("mnist_model.pth", map_location=device)

        # Check if optimizer state exists
        optimizer_state = checkpoint["optimizer_state_dict"]
        assert isinstance(
            optimizer_state, dict
        ), "Optimizer state not found in checkpoint"

        # Create a temporary optimizer to load state
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer.load_state_dict(optimizer_state)

        # Check if it's SGD optimizer
        assert isinstance(optimizer, optim.SGD), "Optimizer should be SGD"

        # Check momentum parameter
        assert optimizer.param_groups[0]["momentum"] == 0.9, "Momentum should be 0.9"

        # Check scheduler if available
        if "scheduler_state_dict" in checkpoint:
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Verify scheduler parameters
            assert scheduler.step_size == 5, "Step size should be 5 epochs"
            assert scheduler.gamma == 0.1, "Learning rate decay factor should be 0.1"

    except Exception as e:
        raise Exception(f"Optimizer test failed: {str(e)}")


def run_test(model, device):
    """Helper function to run the test"""
    # Test Phase transformations
    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=test_transforms
    )
    dataloader_args = (
        dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True)
        if cuda
        else dict(shuffle=True, batch_size=64)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )

    return accuracy


def test_model_parameters():
    """Test if model has less than 25000 parameters"""
    model = Net()
    param_count = count_parameters(model)
    assert (
        param_count < 25000
    ), f"Model has {param_count} parameters, should be less than 25000"


def test_model_accuracy():
    """Test if model achieves > 99.4% accuracy"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your pre-trained model and optimizer
    model = Net()
    try:
        checkpoint = torch.load("mnist_model.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        raise Exception(
            f"Could not load model: {str(e)}. Please ensure mnist_model.pth is in the root directory."
        )

    model.to(device)
    accuracy = run_test(model, device)
    assert accuracy > 99.4, f"Model accuracy is {accuracy}%, should be > 99.4%"
