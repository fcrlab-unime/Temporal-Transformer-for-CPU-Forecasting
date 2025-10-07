import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from ForecastingTransformer import ForecastingTransformer
from VMCPUDataset import VMCPUDataset
from helper import * 


# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    # ---- Config ----
    num_vms = 100
    batch_size = 96
    num_epochs = 1000
    seq_len =  24
    pred_len = 3
    vm_indices = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]  # VM to plot (0-based)
    dataset_name = 'alibaba'  # for title only
    model_save_path = f'./models/{seq_len}/{seq_len}-{pred_len}-transfer.pth'
    model_type = "VMs" if not 'transfer' in model_save_path else "VMs + Containers"

    parser = argparse.ArgumentParser(description="Forecasting Transformer Training")
    parser.add_argument('--test', action='store_true', help='Run in test mode with reduced settings')
    parser.add_argument('--transfer', action='store_true', help='Run in training mode')
    args = parser.parse_args()

    # ---- Data ----
    dataset = VMCPUDataset(csv_dir='./dataset', seq_len=seq_len, pred_len=pred_len, prefix=dataset_name, scaling='minmax')
    dataset_name = dataset_name.capitalize()  # for title only

    total_length = len(dataset)
    gap = max(1, seq_len + pred_len - 1) if not args.test else 0
    train_range, val_range, test_range = split_with_gaps(total_length, 0.70, 0.15, gap=gap)

    train_dataset = Subset(dataset, train_range)
    val_dataset = Subset(dataset, val_range)
    test_dataset = Subset(dataset, test_range)

    print(f"Splits -> train [{train_range.start},{train_range.stop}) | "
          f"val [{val_range.start},{val_range.stop}) | "
          f"test [{test_range.start},{test_range.stop}) | gap={gap}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # ---- Model ----
    model = ForecastingTransformer(
        num_vms=100,
        num_features=1,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=192,
        num_heads_spatial=0,      # d_head = 32
        num_heads_temporal=8,     # or 4–6 if you prefer larger d_head
        ff_dim=256,
        num_layers_spatial=0,
        num_layers_temporal=3,
        dropout=0.15,             # slightly lower if underfitting; raise to 0.2–0.3 if overfitting
        use_vm_embed=True,
        use_time_embed=True,
        grad_ckpt=True,           # checkpoint encoders to save VRAM
    )


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Train or load model ----
    if args.test or args.transfer:
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print("Loaded pre-trained model state.")
        except FileNotFoundError:
            print("No pre-trained model found, please train the model first.")
            exit(1)

    if not args.test:
        train_out = train_model(
                model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=num_epochs, patience=50, lr_decay_factor=0.5, min_delta=1e-6, min_lr=1e-6)

    # ---- Evaluate ----
    # Provide test_dataset so test_model can invert per VM for original-scale metrics

    metrics = test_model(model, test_loader, device, dataset=test_dataset)

    if args.test: 
        for vm_index in vm_indices:
            # ---- Build global series for one VM across full dataset ----
            # Get per-VM ground truth series and predictions, with per-VM inverse to original units
            y_train_vm = collect_vm_series_from_subset(train_dataset, vm_index=vm_index, invert=True)
            y_val_vm = collect_vm_series_from_subset(val_dataset, vm_index=vm_index, invert=True)
            y_test_vm_true = collect_vm_series_from_subset(test_dataset, vm_index=vm_index, invert=True)
            full_series = np.concatenate([y_train_vm, y_val_vm, y_test_vm_true], axis=0)

            # Predictions on test subset for the chosen VM (aligned to test indices), inverted to original units
            y_test_vm_pred = predict_vm_series_on_subset(model, test_dataset, device, vm_index=vm_index, invert=True)

            # ---- Time index and boundaries ----
            time_index = np.arange(len(full_series))  # or seconds if available
            train_end_idx = len(y_train_vm)  # exclusive end
            val_end_idx = train_end_idx + len(y_val_vm)
            test_start_idx = len(y_train_vm) + len(y_val_vm)

            """ plot_test_set_only(
                time_index=time_index,
                full_series=full_series,           # original units
                test_start_idx=test_start_idx,
                pred_series=y_test_vm_pred,        # original units
                title=f"Model={model_type} | Lookback={seq_len} | Predict={pred_len} | Dataset={dataset_name}",
                vm_index=vm_index,
                save_path=f"cpu_usage_vm{vm_index+1}_test_only.pdf"
            )
            """
            pred_train_series = predict_vm_series_on_subset(model, train_dataset, device, vm_index=vm_index, invert=True)
            pred_val_series = predict_vm_series_on_subset(model, val_dataset, device, vm_index=vm_index, invert=True)
            pred_test_series = predict_vm_series_on_subset(model, test_dataset, device, vm_index=vm_index, invert=True)

            """ 
            plot_full_continuous_prediction(
                model=model,
                dataset=dataset,  # not just test_dataset: the full dataset
                device=device,
                train_end_idx=len(train_dataset),
                val_end_idx=len(train_dataset) + len(val_dataset),
                test_start_idx=len(train_dataset) + len(val_dataset),
                vm_index=vm_index,
                invert=True,
                title=f"Model={model_type} | Lookback={seq_len} | Predict={pred_len} | Dataset={dataset_name}",
                save_path=f"cpu_usage_vm{vm_index+1}_continuous.pdf"
            )
            """
            
            plot_input_output_window(
                model=model,
                dataset=val_dataset,   # oppure test_dataset
                device=device,
                sample_idx=0,          # quale finestra vuoi plottare
                vm_index=vm_index,            # quale VM vuoi plottare (indice)
                seq_len=seq_len,            # lunghezza finestra input
                pred_len=pred_len,           # lunghezza predizione
                title=f"Model={model_type} | Lookback={seq_len} | Predict={pred_len} | Dataset={dataset_name} | VM={vm_index+1}",
                save_path=f"window_{vm_index}.pdf"
            )


#avg_time, per_vm_times, ci_low, ci_high = measure_inference_time(model, test_dataset, device, range(10))
#print(f"Average inference time over 10 VMs: {avg_time:.6f} seconds")
#print(f"95% confidence interval: [{ci_low:.6f}, {ci_high:.6f}] seconds")