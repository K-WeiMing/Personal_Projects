# Import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, batch_size=32)
# MNIST input: 1, 28, 28


# Image Classifier Neural Network
class ImaageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # 1 channel, 32 kernels of shape 3x3
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            # Notes: Each Conv2D reduces image size by 2 pixels,
            # Resulting output = 64 * (28-6) * (28-6)
            # Where 28 refers to the starting input dimensions
            nn.Linear(64 * (28 - 6) * (28 - 6), 10),
        )

    def forward(self, x):
        return self.model(x)


# Instance of the neural network, loss, optimizer
clf = ImaageClassifier().to("cpu")
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


# Training flow
def train_model():
    for epoch in range(10):  # Train for 10 epochs
        for batch in dataset:
            X, y = batch
            X, y = X.to("cpu"), y.to("cpu")
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backpropogation
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch: {epoch} loss is: {loss.item()}")

    with open("model_sate_pt", "wb") as f:
        save(clf.state_dict(), f)


# Predict
def predict_image():
    with open("model_state.pt", "rb") as f:
        clf.load_state_dict(load(f))

        img = Image.open("img_3.jpg")
        img_tensor = ToTensor()(img).unsqueeze(0).to("cpu")
        print(torch.argmax(clf(img_tensor)))


if __name__ == "__main__":
    # train_model()
    predict_image()
