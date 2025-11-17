import pickle
import numpy as np
import matplotlib.pyplot as plt
import CNN

def visualize_filters(model):
    # ACCESS FILTER VALUES CORRECTLY
    filters = model.conv1.W["val"]   # NOT model.conv1.W

    print("Conv1 filters shape:", filters.shape)
    # Expected shape: (6, 1, 3, 3)

    num_filters = filters.shape[0]

    plt.figure(figsize=(10, 5))
    for i in range(num_filters):
        f = filters[i, 0]  # take channel 0 (grayscale)
        plt.subplot(2, 3, i+1)
        plt.imshow(f, cmap="gray")
        plt.title(f"Filter {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("lenet5_conv1_filters.png")
    print("Saved: lenet5_conv1_filters.png")
    plt.show()


if __name__ == "__main__":
    # Load saved weights
    with open("weights_LeNet5_SGD.pkl", "rb") as f:
        params = pickle.load(f)

    model = CNN.LeNet5()
    model.set_params(params)

    visualize_filters(model)
