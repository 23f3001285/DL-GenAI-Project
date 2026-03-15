import torch
from transformers import ASTForAudioClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=10
    ).to(DEVICE)

    print("AST model ready")

if __name__ == "__main__":
    main()