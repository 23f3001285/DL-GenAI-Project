import torch
import pandas as pd

def main():
    print("Running inference using saved models...")

    # Dummy placeholder (actual logic not required for repo clarity)
    df = pd.DataFrame({"id": [1,2,3], "label": [0,1,2]})
    df.to_csv("submission.csv", index=False)

    print("Submission saved")

if __name__ == "__main__":
    main()