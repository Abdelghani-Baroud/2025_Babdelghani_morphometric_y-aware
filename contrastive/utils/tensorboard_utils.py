import matplotlib.pyplot as plt
import io
from torchvision.transforms.functional import to_tensor
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def log_rbf_kernel_heatmap(writer: SummaryWriter, weights: np.ndarray, step: int, tag: str = "rbf_kernel"):
    """Logs a heatmap of the RBF kernel weights and statistics to TensorBoard."""
    try:
        if weights.size == 0:
            print("[WARNING] Empty RBF kernel weights at step", step)
            return

        fig, ax = plt.subplots(figsize=(5, 4))
        cax = ax.imshow(weights, cmap='viridis', aspect='auto')
        ax.set_title("RBF Kernel Similarity")
        fig.colorbar(cax)
        plt.tight_layout()

        # Save figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        

        # DEBUG: Save image locally to check it's valid
        debug_path = f"/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/contrastive//kernel_heatmap_before_normalization/rbf_debug_step_sigma0.001_normmalized_{step}.png"
        plt.savefig(debug_path, dpi=100)
        print(f"[DEBUG] Saved RBF kernel image to {debug_path}")


        # Read image and convert to tensor
        image = plt.imread(buf)[..., :3]  # Drop alpha channel if present
        image_tensor = to_tensor(image.astype(np.float32))  # Ensure float32

        writer.add_image(tag, image_tensor, step)

        # Log scalar stats
        mean_val = weights.mean()
        max_val = weights.max()
        min_val = weights.min()

        writer.add_scalar(f"{tag}/mean_similarity", mean_val, step)
        writer.add_scalar(f"{tag}/max_similarity", max_val, step)
        writer.add_scalar(f"{tag}/min_similarity", min_val, step)

        buf.close()
        plt.close(fig)

    except Exception as e:
        print(f"[WARNING] Failed to log RBF kernel image or stats at step {step}: {e}")
