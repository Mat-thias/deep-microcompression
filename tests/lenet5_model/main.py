import torch
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms

import sys

# sys.path.append("/home/matthias/Documents/EmbeddedAI/deep-microcompression/")
sys.path.append("../../")

from development import (
    Sequential,
    Conv2d,
    Linear,
    ReLU,
    MaxPool2d,
    Flatten
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
lenet5_file = f"lenet5_state_dict_{DEVICE}.pth"

LUCKY_NUMBER = 25
torch.manual_seed(LUCKY_NUMBER)
torch.random.manual_seed(LUCKY_NUMBER)
torch.cuda.manual_seed(LUCKY_NUMBER)

data_transform = transforms.Compose([
    transforms.RandomCrop((24, 24)),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

mnist_train_dataset = datasets.MNIST("./datasets", train=True, download=True, transform=data_transform)
mnist_test_dataset = datasets.MNIST("./datasets", train=False, download=True, transform=data_transform)

mnist_train_loader = data.DataLoader(mnist_train_dataset, batch_size=32, shuffle=True)
mnist_test_loader = data.DataLoader(mnist_test_dataset, batch_size=32)


lenet5_model = Sequential(
    Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2, padding=0),

    Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2, padding=0),

    Flatten(),

    Linear(in_features=16*5*5, out_features=84, bias=True),
    ReLU(),
    Linear(in_features=84, out_features=10, bias=True)
).to(DEVICE)

accuracy_fun = lambda y_pred, y_true: (y_pred.argmax(dim=1) == y_true).sum().item()

try:
    # raise RuntimeError
    lenet5_model.load_state_dict(torch.load(lenet5_file, weights_only=True))
    
except (RuntimeError, FileNotFoundError):

    criterion_fun = nn.CrossEntropyLoss()
    optimizion_fun = optim.Adam(lenet5_model.parameters(), lr=1.e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizion_fun, mode="min", patience=2)

    lenet5_model.fit(
        mnist_train_loader, 15, 
        criterion_fun, optimizion_fun, lr_scheduler,
        validation_dataloader=mnist_test_loader, 
        device=DEVICE
    )
    torch.save(lenet5_model.state_dict(), lenet5_file)

original_acc = lenet5_model.evaluate(mnist_test_loader, accuracy_fun, device=DEVICE)
print(f"The original model accuracy is {original_acc*100:.2f}%.")

lenet5_model.cpu()

# PRUNED MODEL
pruned_sparsity = [i/10 for i in range(10)]
for sparsity in pruned_sparsity:
    pruned_model = lenet5_model.prune_channel(sparsity)
    acc = pruned_model.evaluate(mnist_test_loader, accuracy_fun)
    print(f"The pruned model with sparsity {sparsity} accuracy is {acc*100:.2f}%.")
    print(f"The accurancy drop is {(original_acc - acc)*100:.2f}%")

quantization_bitwidth = [i for i in range(8, 0, -1)]

# DYNAMIC QUANTIZED MODEL PER TERSON
for bitwidth in quantization_bitwidth:
    dynamic_quantized_per_tensor_model = lenet5_model.dynamic_quantize_per_tensor(bitwidth)
    acc = dynamic_quantized_per_tensor_model.evaluate(mnist_test_loader, accuracy_fun)
    print(f"The dynamic quantized per tensor model with bitwidth {bitwidth} accuracy is {acc*100:.2f}%.")
    print(f"The accurancy drop is {(original_acc - acc)*100:.2f}%")


# DYNAMIC QUANTIZED MODEL PER TERSON
for bitwidth in quantization_bitwidth:
    dynamic_quantized_per_channel_model = lenet5_model.dynamic_quantize_per_channel(bitwidth)
    acc = dynamic_quantized_per_channel_model.evaluate(mnist_test_loader, accuracy_fun)
    print(f"The dynamic quantized per channel model with bitwidth {bitwidth} accuracy is {acc*100:.2f}%.")
    print(f"The accurancy drop is {(original_acc - acc)*100:.2f}%")


# STATIC QUANTIZED MODEL PER TERSON
for bitwidth in quantization_bitwidth:
    static_quantized_per_tensor_model = lenet5_model.static_quantize_per_tensor(next(iter(mnist_test_loader))[0], bitwidth)
    acc = static_quantized_per_tensor_model.evaluate(mnist_test_loader, accuracy_fun)
    print(f"The static quantized per tensor model with bitwidth {bitwidth} accuracy is {acc*100:.2f}%.")
    print(f"The accurancy drop is {(original_acc - acc)*100:.2f}%")


# STATIC QUANTIZED MODEL PER TERSON
for bitwidth in quantization_bitwidth:
    static_quantized_per_channel_model = lenet5_model.static_quantize_per_channel(next(iter(mnist_test_loader))[0], bitwidth)
    acc = static_quantized_per_channel_model.evaluate(mnist_test_loader, accuracy_fun)
    print(f"The static quantized per channel model with bitwidth {bitwidth} accuracy is {acc*100:.2f}%.")
    print(f"The accurancy drop is {(original_acc - acc)*100:.2f}%")

