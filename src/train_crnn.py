import os
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ImprovedCRNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(64 * 32, 128, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, -1)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.fc(x)

def main():
    model = ImprovedCRNN().to(DEVICE)
    print("CRNN ready (training code omitted for repo cleanliness)")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/CRNN_best.pt")

if __name__ == "__main__":
    main()