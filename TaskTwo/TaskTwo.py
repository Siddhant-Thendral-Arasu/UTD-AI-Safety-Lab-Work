from torchvision.datasets import MNIST
from torchvision.datasets import EMNIST
print("MNIST dataset imported successfully.")
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from TaskTwoInfo.lenet import LeNet
from tqdm import tqdm
from TaskTwoInfo.losses import edl_mse_loss
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from TaskTwoInfo.losses import softplus_evidence
from TaskTwoInfo.helpers import one_hot_embedding

transform = transforms.Compose([transforms.ToTensor()])

mnist_train = MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = MNIST(root='./data', train=False, transform=transform, download=True)
print(len(mnist_train))
print(len(mnist_test))

def edl_entropy(alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)  # Total belief mass
    dirichlet_entropy = torch.sum(
        (alpha / S) * (torch.digamma(S) - torch.digamma(alpha)),
        dim=1
    )
    return dirichlet_entropy

mnist_train_loader = DataLoader(mnist_train, batch_size=1000, shuffle=True, num_workers=8)
mnist_test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False, num_workers=8)
print("DataLoader created successfully.")

mnist_dataloaders = {
    "train": mnist_train_loader,
    "test": mnist_test_loader
}

emnist_test = EMNIST(root='./data', split='letters', train=False, transform=transform, download=True)
emnist_test_loader = DataLoader(emnist_test, batch_size=1000, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LeNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for epoch in range(15):
    print("Epoch {}/{}".format(epoch, 14))
    print("-" * 10)

    # Each epoch has a training and validation phase
    for phase in ["train", "test"]:
        if phase == "train":
            print("Training...")
            model.train()  # Set model to training mode
        else:
            print("Validating...")
            model.eval()  # Set model to evaluate mode

        for i, (inputs, labels) in enumerate(mnist_dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                y = one_hot_embedding(labels, 10)
                y = y.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = edl_mse_loss(outputs, y.float(), epoch, 10, 10, device)
                
                if phase == "train":
                    loss.backward()
                    optimizer.step()
            
        if exp_lr_scheduler is not None:
            if phase == "train":
                exp_lr_scheduler.step()


model.eval()  # Set model to evaluation mode
entropies = []

with torch.no_grad():
    for i, (images, _) in enumerate(emnist_test_loader):
        images = images.to(device)
        
        # Forward pass
        logits = model(images)
        evidence = softplus_evidence(logits)
        alpha = evidence + 1

        # Compute EDL entropies
        batch_entropies = edl_entropy(alpha).cpu().numpy()
        entropies.append(batch_entropies)
        
        # Optional: Progress tracking
        if i % 100 == 0:
            print(f"Processed {i} batches.")

entropies = np.concatenate(entropies)
sorted_entropies = np.sort(entropies)
probabilities = np.arange(1, len(sorted_entropies) + 1) / len(sorted_entropies)

plt.plot(sorted_entropies, probabilities, label='EMNIST (Out-of-Distribution)', linestyle='--', color='red')
plt.xlabel("Entropy")
plt.ylabel("Probability")
plt.title("Empirical CDF of Entropy (OOD)")
plt.legend()
plt.grid()

save_path = f"entropy_cdf_graph.png"
plt.savefig(save_path)
print(f"Graph saved at: {save_path}")





