import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch


class EarlyStopping:
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.save_path)	
        self.val_loss_min = val_loss

class ActivationVisualizer:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self._register_hooks(model)

    def _register_hooks(self, module, prefix=''):
        # Register hooks to capture the activations of each layer
        for name, layer in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name  # For nested layers
            if isinstance(layer, nn.Conv3d):
                layer.register_forward_hook(self._hook(full_name))
            elif len(list(layer.children())) > 0:  # If the layer contains sub-layers, recurse
                self._register_hooks(layer, prefix=full_name)

    def _hook(self, layer_name):
        def hook_fn(module, input, output):
            self.activations[layer_name] = output.clone().detach()

        return hook_fn

    def get_activation(self, layer_name):
        if layer_name not in self.activations:
            raise ValueError(f"Activation for layer '{layer_name}' is not available.")
        return self.activations[layer_name]

    def plot_activation_maps(self, activation, num_filters=4):
        activation = activation[0, :num_filters, :, :, :].cpu().numpy()

        fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
        for i, ax in enumerate(axes):
            # mid_x = activation[i, activation.shape[1] // 2, :, :]
            # mid_y = activation[i, :, activation.shape[2] // 2, :]
            mid_z = activation[i, :, :, activation.shape[3] // 2]
            ax.imshow(mid_z, cmap='viridis')
            ax.axis('off')
            ax.set_title(f"Filter {i + 1}")
        plt.tight_layout()
        plt.show()


class Normalizer:
    def normalize(self, tensor):
        return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

class ParameterCounter:
    def count_parameters(self,model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

class LossStabilityCalculator:
    def calculate_stability(losses, loss_type):
        num_epochs_to_consider = 10
        final_losses = losses[-num_epochs_to_consider:]
        average_loss = np.mean(final_losses)
        std_deviation = np.std(final_losses)
        print(f'{loss_type} Loss Stability: {average_loss:.4f} ± {std_deviation:.4f}')

