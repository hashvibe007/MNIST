import torch
import torchvision
import torch.nn.functional as F
import sys
import os
from torchvision import datasets, transforms
from model import Net

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

    # Load your pre-trained model
    model = Net()
    try:
        model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
    except:
        raise Exception(
            "Could not find mnist_model.pth. Please ensure the model is saved in the root directory."
        )

    model.to(device)

    # Run test and get accuracy
    accuracy = run_test(model, device)

    assert accuracy > 99.4, f"Model accuracy is {accuracy}%, should be > 99.4%"
