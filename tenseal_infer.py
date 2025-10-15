import torch
import torch.nn as nn
import tenseal as ts

# ------------------------------
# Define the same architecture
# ------------------------------
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

# ------------------------------
# Load model with checkpoint
# ------------------------------
def load_model(input_dim_for_model, model_path="lung_cancer_model.pth"):
    saved_model = DeepNN(input_dim_for_model)

    try:
        checkpoint = torch.load(model_path, map_location="cpu")

        # fix: remove "model." prefix if present
        if any(k.startswith("model.") for k in checkpoint.keys()):
            checkpoint = {k.replace("model.", ""): v for k, v in checkpoint.items()}
            saved_model.model.load_state_dict(checkpoint)
        else:
            saved_model.load_state_dict(checkpoint)

        print(f"✅ Model loaded successfully from {model_path}")

    except FileNotFoundError:
        print(f"⚠️ {model_path} not found. Initializing random weights.")
        for param in saved_model.parameters():
            if param.dim() > 1:
                nn.init.kaiming_uniform_(param, nonlinearity="relu")
            else:
                nn.init.constant_(param, 0.0)
    except Exception as e:
        print(f"⚠️ Error loading model: {e}. Using random weights.")

    saved_model.eval()
    return saved_model

# ------------------------------
# Example TenSEAL inference
# ------------------------------
if __name__ == "__main__":
    input_dim = 15   # <-- change this to match your dataset
    model = load_model(input_dim)

    # Create TenSEAL CKKS context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 21, 21, 40]
    )
    context.global_scale = 2**21
    context.generate_galois_keys()

    # Example input vector (encrypted)
    sample_input = [0.1] * input_dim
    enc_input = ts.ckks_vector(context, sample_input)

    # Decrypt before feeding into torch model (since torch doesn’t accept encrypted tensors)
    # In real encrypted inference, you’d need a homomorphic-friendly model
    dec_input = torch.tensor(enc_input.decrypt(), dtype=torch.float32)
    output = model(dec_input)

    print("Model output:", output.item())
