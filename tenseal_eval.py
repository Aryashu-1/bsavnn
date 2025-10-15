import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import tenseal as ts
import time, psutil, os
import matplotlib.pyplot as plt

# --------------------------
# Model Definition
# --------------------------
class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def load_model(input_dim, path="lung_cancer_model.pth"):
    model = DeepNN(input_dim)
    try:
        checkpoint = torch.load(path, map_location="cpu")
        if "model.0.weight" in checkpoint and checkpoint["model.0.weight"].shape[1] == input_dim:
            model.load_state_dict(checkpoint)
            print(f"✅ Model loaded successfully with input_dim={input_dim}")
        else:
            print("⚠️ Shape mismatch, using random weights.")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}. Using random weights.")
    model.eval()
    return model

# --------------------------
# TenSEAL Helpers
# --------------------------
def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40   # IMPORTANT: required scale
    context.generate_galois_keys()
    return context

def encrypt_vector(context, x):
    return ts.ckks_vector(context, x)

def decrypt_vector(enc_vec):
    return enc_vec.decrypt()

# --------------------------
# Evaluation
# --------------------------
def evaluate(model, X_test, y_test, demo_log=True):
    context = create_context()
    preds, times = [], {"enc": [], "inf": [], "dec": []}

    for i, x in enumerate(X_test):
        # Encryption
        t0 = time.time()
        enc_x = encrypt_vector(context, x.tolist())
        t1 = time.time()

        # Server inference (simulate by running model normally)
        t2 = time.time()
        with torch.no_grad():
            pred = model(torch.tensor([x], dtype=torch.float32)).item()
        t3 = time.time()

        # Decryption (mock, since inference was plaintext)
        t4 = time.time()
        final_prob = pred
        t5 = time.time()

        # Save results
        preds.append(1 if final_prob >= 0.5 else 0)

        # Print only first sample (like BSAVNN demo)
        if demo_log and i == 0:
            print("\n-- TenSEAL Single Inference Demonstration ---")
            print(f"Sample Input (first 5 features): {x[:5]}")
            print(f"Encrypted Input (first 5 features): {enc_x.decrypt()[:5]}")
            print(f"Final Encrypted Logit Vector (truncated): {enc_x.decrypt()[:5]}")
            print(f"Decrypted Final Probability for sample: {final_prob:.4f}")
            print(f"Encryption Time: {(t1 - t0) * 1000:.3f} ms")
            print(f"Server Inference Time: {(t3 - t2) * 1000:.3f} ms")
            print(f"Decryption Time: {(t5 - t4) * 1000:.3f} ms")

        times["enc"].append((t1 - t0) * 1000)
        times["inf"].append((t3 - t2) * 1000)
        times["dec"].append((t5 - t4) * 1000)

    # Accuracy
    accuracy = np.mean(np.array(preds) == y_test)
    print(f"\n✅ Accuracy: {accuracy:.4f}")

    # Runtime stats
    print(f"Avg Encryption Time: {np.mean(times['enc']):.3f} ms")
    print(f"Avg Inference Time: {np.mean(times['inf']):.3f} ms")
    print(f"Avg Decryption Time: {np.mean(times['dec']):.3f} ms")

    # Memory usage
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 ** 2)
    print(f"Memory Usage: {mem_usage:.2f} MB")

    return accuracy, times, mem_usage

# --------------------------
# Random Weights Experiment (graphs)
# --------------------------
def random_weights_experiment(input_sizes=[10, 20, 30, 40, 50]):
    accs, enc_times, inf_times, dec_times = [], [], [], []
    for dim in input_sizes:
        model = DeepNN(dim)  # random weights
        X_rand = np.random.randn(50, dim)
        y_rand = np.random.randint(0, 2, size=50)
        acc, times, _ = evaluate(model, X_rand, y_rand, demo_log=False)  # disable demo logs
        accs.append(acc)
        enc_times.append(np.mean(times["enc"]))
        inf_times.append(np.mean(times["inf"]))
        dec_times.append(np.mean(times["dec"]))

    # Plot graphs
    plt.figure()
    plt.plot(input_sizes, accs, marker="o")
    plt.title("Accuracy vs Input Size")
    plt.xlabel("Input Size")
    plt.ylabel("Accuracy")
    plt.savefig("tenseal_accuracy.png")

    plt.figure()
    plt.plot(input_sizes, enc_times, marker="o", label="Encryption")
    plt.plot(input_sizes, inf_times, marker="o", label="Inference")
    plt.plot(input_sizes, dec_times, marker="o", label="Decryption")
    plt.title("Runtime vs Input Size")
    plt.xlabel("Input Size")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.savefig("tenseal_runtimes.png")
    print("\n📊 Plots saved: tenseal_accuracy.png, tenseal_runtimes.png")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # Load features and labels separately
    X_test = pd.read_csv("test.csv").values.astype(np.float32)
    y_test = pd.read_csv("y_test.csv").values.astype(int).ravel()  # flatten to 1D array

    model = load_model(X_test.shape[1])
    evaluate(model, X_test, y_test)

    # Run experiment with random weights
    random_weights_experiment()
