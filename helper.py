import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import scipy.stats
import numpy as np
import argparse
import time
from ForecastingTransformer import ForecastingTransformer
from VMCPUDataset import VMCPUDataset

def plot_full_with_train_and_pred(time_index, full_series, train_end_idx, test_start_idx,
                                  pred_series, title="Mean CPU usage", vm_index=0,
                                  save_path="vm_plot.pdf"):
    """
    Assumes full_series and pred_series are already in desired units (scaled or original).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_index, full_series, color='#1f77b4', linewidth=0.8, label='Mean CPU usage data')
    ax.plot(time_index[:train_end_idx], full_series[:train_end_idx],
            color='#ff7f0e', linewidth=1.2, label='Training')
    ax.plot(time_index[test_start_idx:test_start_idx + len(pred_series)],
            pred_series, color='#2ca02c', linewidth=1.2, label='Prediction')

    if train_end_idx > 0:
        ax.axvline(time_index[train_end_idx - 1], color='k', linestyle='--', linewidth=0.8)
    ax.axvline(time_index[test_start_idx], color='k', linestyle='--', linewidth=0.8)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean CPU usage")
    ax.set_title(f"{title} | VM {vm_index+1}")
    ax.legend(loc='upper left')
    ax.margins(x=0)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


    # ----------------------
# Helpers
# ----------------------
def split_with_gaps(total_len, train_ratio=0.70, val_ratio=0.15, gap=1):
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)

    train_start = 0
    train_end = train_len

    val_start = min(train_end + gap, total_len)
    val_end = min(val_start + val_len, total_len)

    test_start = min(val_end, total_len)
    test_end = total_len
    return range(train_start, train_end), range(val_start, val_end), range(test_start, test_end)

def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}

def flatten_preds_targets(outputs, targets):
    if outputs.ndim == 4 and outputs.size(1) == 1:
        outputs = outputs.squeeze(1)  # [B, num_vms, num_features]
    if targets.ndim == 4 and targets.size(1) == 1:
        targets = targets.squeeze(1)
    return outputs.reshape(-1).cpu().numpy(), targets.reshape(-1).cpu().numpy()

# ----------------------
# Train / Eval
# ----------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=50, patience=100, lr_decay_factor=0.5, min_delta=0.0, min_lr=1e-6):
    model.to(device)
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_decay_factor,
        patience=max(1, patience // 2), min_lr=min_lr
    )
    """

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            if y.ndim == 3:
                y = y.unsqueeze(1)  # [B, 1, num_vms, num_features]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # ---- Validate ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                if y.ndim == 3:
                    y = y.unsqueeze(1)
                outputs = model(x)
                val_loss = criterion(outputs, y)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / max(1, len(val_loader))

        #scheduler.step(avg_val_loss)

        group_lrs = [g["lr"] for g in optimizer.param_groups]
        current_lr = group_lrs if len(group_lrs) == 1 else group_lrs

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)
        print(f"Epoch [{epoch+1}/{num_epochs}] | train_loss: {avg_train_loss:.6f} | val_loss: {avg_val_loss:.6f} | lr: {current_lr}")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), "best_model.pth")
        """ else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping: no val improvement in {patience} epochs. Best val_loss: {best_val_loss:.6f}")
                break """

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_loss": best_val_loss, "history": history}

def test_model(model, dataloader, device, dataset=None):
    """
    Evaluate the model on a dataloader.
    Returns both scaled-space metrics and per-VM original-unit metrics if dataset is provided.
    The dataset must implement inverse_transform_vm(vm_idx, arr_scaled_1d).
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)  # [B, pred_len, num_vms, 1]

            # Flatten over pred_len and VMs
            yp, yt = flatten_preds_targets(outputs, y)  # both 1D arrays
            all_preds.append(yp)
            all_targets.append(yt)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    scaled_metrics = compute_metrics(y_true, y_pred)
    print(f"[Scaled] R2: {scaled_metrics['r2']:.4f}, MAE: {scaled_metrics['mae']:.4f}, "
          f"MSE: {scaled_metrics['mse']:.4f}, RMSE: {scaled_metrics['rmse']:.4f}")

    original_metrics = None
    if dataset is not None:
        vm_true = [[] for _ in range(dataset.dataset.num_vms)]
        vm_pred = [[] for _ in range(dataset.dataset.num_vms)]

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)

                # Move to CPU and remove last singleton dimension
                y_np = y.squeeze(-1).cpu().numpy()        # [B, pred_len, num_vms]
                o_np = outputs.squeeze(-1).cpu().numpy()  # [B, pred_len, num_vms]

                # Append per VM (flatten pred_len)
                for vm_idx in range(dataset.dataset.num_vms):
                    vm_true[vm_idx].extend(y_np[:, :, vm_idx].reshape(-1).tolist())
                    vm_pred[vm_idx].extend(o_np[:, :, vm_idx].reshape(-1).tolist())

        true_inv_all = []
        pred_inv_all = []
        for vm_idx in range(dataset.dataset.num_vms):
            t = np.array(vm_true[vm_idx], dtype=np.float32)
            p = np.array(vm_pred[vm_idx], dtype=np.float32)
            true_inv_all.append(dataset.dataset.inverse_transform_vm(vm_idx, t))
            pred_inv_all.append(dataset.dataset.inverse_transform_vm(vm_idx, p))

        y_true_inv = np.concatenate(true_inv_all)
        y_pred_inv = np.concatenate(pred_inv_all)
        original_metrics = compute_metrics(y_true_inv, y_pred_inv)

    return {"scaled": scaled_metrics, "original": original_metrics}


# ----------------------
# Plot like the reference
# ----------------------
def collect_vm_series_from_subset(ds_subset, vm_index=0, invert=False):
    """
    Return concatenated ground-truth series (for that VM) across the subset order.
    Works for any pred_len.
    """
    base = ds_subset.dataset
    idxs = list(ds_subset.indices)
    ys = []
    for i in idxs:
        _, y = base[i]  # y shape [pred_len, num_vms, 1]
        y = y.squeeze(-1)  # [pred_len, num_vms]
        ys.extend(y[:, vm_index].tolist())  # append all pred_len steps for this VM
    series = np.array(ys, dtype=np.float32)
    if invert:
        series = base.inverse_transform_vm(vm_index, series)
    return series


def predict_vm_series_on_subset(model, ds_subset, device, vm_index=0, invert=False):
    """
    Return concatenated predictions for that VM across the subset order.
    Works for any pred_len.
    """
    base = ds_subset.dataset
    idxs = list(ds_subset.indices)
    preds = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i in idxs:
            x, _ = base[i]  # x: [seq_len, num_vms, 1]
            x = x.unsqueeze(0).to(device)  # [1, seq_len, num_vms, 1]
            out = model(x)  # [1, pred_len, num_vms, 1]
            out = out.squeeze(0).squeeze(-1)  # [pred_len, num_vms]
            preds.extend(out[:, vm_index].tolist())
    pred_series = np.array(preds, dtype=np.float32)
    if invert:
        pred_series = base.inverse_transform_vm(vm_index, pred_series)
    return pred_series



def plot_full_with_val_test_pred(
    time_index, full_series,
    train_end_idx, val_end_idx, test_start_idx,
    pred_train_series, pred_val_series, pred_test_series,
    title="Mean CPU usage", vm_index=0,
    save_path="vm_plot_val_test_pred.pdf"
):
    """
    Plots ground truth and model predictions for training, validation, and test sections.
    Predictions are aligned to the full series length, with NaN outside their segment.
    """
    n = len(full_series)

    # Initialize NaN arrays the same length as full_series
    pred_train_full = np.full(n, np.nan, dtype=np.float32)
    pred_val_full   = np.full(n, np.nan, dtype=np.float32)
    pred_test_full  = np.full(n, np.nan, dtype=np.float32)

    # Fill in predictions at the correct absolute positions
    pred_train_full[:train_end_idx] = pred_train_series
    pred_val_full[train_end_idx:val_end_idx] = pred_val_series
    pred_test_full[test_start_idx:test_start_idx+len(pred_test_series)] = pred_test_series

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Ground truth
    ax.plot(time_index, full_series, color='#1f77b4', linewidth=0.8, label='Mean CPU usage data')
    ax.plot(time_index[:train_end_idx], full_series[:train_end_idx],
            color='#ff7f0e', linewidth=1.2, label='Training')

    # Predictions
    ax.plot(time_index, pred_train_full, color='#2ca02c', linewidth=1.2, label='Training Prediction')
    ax.plot(time_index, pred_val_full,   color='#d62728', linewidth=1.2, label='Validation Prediction')
    ax.plot(time_index, pred_test_full,  color='#9467bd', linewidth=1.2, label='Test Prediction')

    # Boundaries
    if train_end_idx > 0:
        ax.axvline(time_index[train_end_idx - 1], color='k', linestyle='--', linewidth=0.8)
    ax.axvline(time_index[val_end_idx - 1], color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(time_index[test_start_idx], color='k', linestyle='--', linewidth=0.8)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean CPU usage")
    ax.set_title(f"{title} | VM {vm_index+1}")
    ax.legend(loc='upper left')
    ax.margins(x=0)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()





def plot_test_set_only(time_index, full_series, test_start_idx, pred_series, 
                      title="Mean CPU usage", vm_index=0, save_path="vm_test_plot.pdf"):
    """
    Plots only the test set and its corresponding predictions.
    Assumes full_series and pred_series are already in desired units (scaled or original).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot only the test set portion from the full series
    ax.plot(time_index[test_start_idx:], full_series[test_start_idx:], 
            color='#1f77b4', linewidth=0.8, label='Test data')

    # Plot corresponding predictions
    ax.plot(time_index[test_start_idx:test_start_idx + len(pred_series)],
            pred_series, color='#2ca02c', linewidth=1.2, label='Prediction')

    # Vertical line at the start of the test set
    ax.axvline(time_index[test_start_idx], color='k', linestyle='--', linewidth=0.8)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean CPU usage")
    ax.set_title(f"{title} | VM {vm_index+1} (Test set only)")
    ax.legend(loc='upper left')
    ax.margins(x=0)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def measure_inference_time(model, dataset, device, vm_indices, repeat=3):
    """
    Measures mean inference time and returns (mean, per-VM times, lower_CI, upper_CI) for 95% CI.
    """
    model.eval()
    times = []
    with torch.no_grad():
        for vm_idx in vm_indices:
            vm_times = []
            for _ in range(repeat):
                start = time.perf_counter()
                _ = predict_vm_series_on_subset(model, dataset, device, vm_index=vm_idx, invert=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                vm_times.append(end - start)
            times.append(np.mean(vm_times))
    mean_time = np.mean(times)
    std_err = scipy.stats.sem(times)
    conf = 0.95
    # For small n, use t-distribution; for large n, z ~ 1.96
    n = len(times)
    dof = n - 1
    t_crit = scipy.stats.t.ppf((1 + conf) / 2., dof) if n > 1 else 0
    ci_half = t_crit * std_err if n > 1 else 0
    lower_ci = mean_time - ci_half
    upper_ci = mean_time + ci_half
    return mean_time, times, lower_ci, upper_ci

def plot_full_continuous_prediction(
    model,
    dataset,
    device,
    train_end_idx,
    val_end_idx,
    test_start_idx,
    vm_index=0,
    invert=True,
    title="Full Curve with Continuous Prediction",
    save_path="vm_full_continuous_pred.pdf"
):
    """
    Runs the model continuously over the entire dataset and plots ground truth
    vs continuous predictions from start to end.

    Ground truth:
        - Train = orange
        - Validation + Test = blue (same color, separated by vertical lines)

    Predictions:
        - green continuous curve
    """

    model.eval()
    preds, ys = [], []

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

    # Iterate sequentially over dataset
    for i in range(len(dataset)):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)  # [1, seq_len, num_vms, features]
        with torch.no_grad():
            out = model(x)

        # --- Normalize output shape to [B, pred_len, num_vms, 1] ---
        if out.ndim == 2:  # [B, num_vms]
            out = out.unsqueeze(1).unsqueeze(-1)
        elif out.ndim == 3:  # [B, num_vms, 1]
            out = out.unsqueeze(1)
        # else: assume already [B, pred_len, num_vms, 1]

        # take only the first-step prediction
        p = out[0, 0, vm_index, 0].detach().cpu().item()
        preds.append(p)

        # ground truth (only first step)
        if y.ndim == 3:  # [pred_len, num_vms, 1]
            y_val = y[0, vm_index, 0].item()
        elif y.ndim == 2:  # [num_vms, 1]
            y_val = y[vm_index, 0].item()
        else:
            y_val = y.item()
        ys.append(y_val)

    ys = np.array(ys, dtype=np.float32)
    preds = np.array(preds, dtype=np.float32)

    # --- Invert scaling if requested ---
    if invert and hasattr(base_dataset, "inverse_transform_vm"):
        ys = base_dataset.inverse_transform_vm(vm_index, ys)
        preds = base_dataset.inverse_transform_vm(vm_index, preds)

    # Clip to valid CPU range
    ys = np.clip(ys, 0, 100)
    preds = np.clip(preds, 0, 100)

    time_index = np.arange(len(ys))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Train ground truth (orange)
    ax.plot(time_index[:train_end_idx], ys[:train_end_idx],
            color='orange', linewidth=0.8, label='Actual (Train)')

    # Validation + Test ground truth (blue)
    ax.plot(time_index[train_end_idx:], ys[train_end_idx:],
            color='#1f77b4', linewidth=0.8, label='Actual (Val+Test)')

    # Prediction (green)
    ax.plot(time_index, preds, color='#2ca02c', linewidth=1.2, label='Prediction')

    # Vertical boundaries
    if train_end_idx > 0:
        ax.axvline(train_end_idx, color='k', linestyle='--', linewidth=0.8, label='Train/Val boundary')
    if val_end_idx > 0:
        ax.axvline(val_end_idx, color='gray', linestyle='--', linewidth=0.8, label='Val/Test boundary')

    ax.set_title(f"{title} | VM {vm_index+1}")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("CPU Usage (%)")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_input_output_window(
    model,
    dataset,
    device,
    sample_idx=0,
    vm_index=0,
    seq_len=48,
    pred_len=12,
    title="Input + Output Window (Continuous)",
    save_path=None
):
    model.eval()
    x, y_true = dataset[sample_idx]  # x shape: (seq_len, num_vms, num_features)
    x = x.unsqueeze(0).to(device)    # add batch dimension
    y_true = y_true.unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = model(x)

    # Seleziona la VM scelta
    input_window = x[0, :, vm_index, 0].cpu().numpy()
    true_window  = y_true[0, :, vm_index, 0].cpu().numpy()
    pred_window  = y_pred[0, :, vm_index, 0].cpu().numpy()

    # Asse temporale
    timesteps_input  = list(range(seq_len))
    timesteps_output = list(range(seq_len, seq_len + pred_len))

    # Predizione continua: inizia dall'ultimo punto dell'input
    pred_window_continuous = [input_window[-1]] + list(pred_window)
    timesteps_pred_cont = list(range(seq_len - 1, seq_len + pred_len))

    plt.figure(figsize=(10, 5))
    # Ground truth: input + output
    plt.plot(timesteps_input + timesteps_output, list(input_window) + list(true_window),
             label="Ground truth (input+output)", color="orange")
    # Predizione: parte dall'ultimo punto dell'input
    plt.plot(timesteps_pred_cont, pred_window_continuous, label="Prediction", color="green", linestyle="--")

    # Linea verticale per separare input e predizione
    plt.axvline(seq_len-1, color="black", linestyle="dashed")
    
    plt.xlabel("Time steps")
    plt.ylabel("CPU usage (%)")
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()