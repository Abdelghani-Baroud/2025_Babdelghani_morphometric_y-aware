import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def wrap_loss_with_kernel_plotting(
    loss_fn,
    base_dir: str,
    dataset_name: str,
    label_file_name: str,
    interval: int = 500
):
    """
    Replaces loss_fn.kernel with a wrapped version that:
      - builds subfolders
          base_dir/
            dataset_name/
              label_file_name/
                sigma_<value>/
      - every `interval` calls will dump a heatmap in that sigma_… folder.
    """
    # 1) build prefix path (no sigma yet)
    prefix_dir = os.path.join(base_dir, dataset_name, label_file_name)
    os.makedirs(prefix_dir, exist_ok=True)

    # 2) keep original kernel
    orig_kernel = loss_fn.kernel

    # 3) attach a step counter
    loss_fn._kernel_step = 0
    loss_fn._kernel_folder = None  # will set after sigma known

    def wrapped_kernel(y1, y2):
        # compute weights
        weights = orig_kernel(y1, y2)

        # bump step
        loss_fn._kernel_step += 1

        # as soon as sigma is known, create its subfolder once
        if loss_fn._kernel_folder is None:
            sigma = loss_fn.sigma
            sub = f"sigma_{sigma}"
            folder = os.path.join(prefix_dir, sub)
            os.makedirs(folder, exist_ok=True)
            loss_fn._kernel_folder = folder

        # every `interval`, dump a heatmap
        if loss_fn._kernel_step % interval == 0:
            arr = weights if isinstance(weights, np.ndarray) \
                  else weights.detach().cpu().numpy()
            plt.figure(figsize=(6,5))
            sns.heatmap(arr, cmap='viridis', cbar=True)
            plt.title(f"σ={loss_fn.sigma}  step={loss_fn._kernel_step}")
            plt.tight_layout()
            fn = os.path.join(
                loss_fn._kernel_folder,
                f"kernel_step{loss_fn._kernel_step}.png"
            )
            plt.savefig(fn)
            plt.close()

        return weights

    # override
    loss_fn.kernel = wrapped_kernel
    return loss_fn