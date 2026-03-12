import torch
import timm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EfficientNetAudio(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=1,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

def main():
    model = EfficientNetAudio().to(DEVICE)
    print("EfficientNet ready")

if __name__ == "__main__":
    main()