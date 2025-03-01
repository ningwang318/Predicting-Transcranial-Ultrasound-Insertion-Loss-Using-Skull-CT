import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from mDPI_Net import ExInceptionNet, CustomDataset
from InceptionNetV3 import InceptionNet
from functions import EarlyStopping, ActivationVisualizer, Normalizer, ParameterCounter, LossStabilityCalculator
from torch.nn.utils import clip_grad_norm_
from scipy.stats import pearsonr

# Importing custom models
from threeD_CNN import Light3DCNN
from inceptionNet import Inception3DCNN
from res_net import generate_model

def set_seed(seed=316):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 在代码开始时调用设置种子函数
set_seed(316)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Data Preparation
data_path = 'E:\\trans_effi\\data\\seg_ct_s_normal_3_2.npz'
with np.load(data_path) as data:
    X = [torch.from_numpy(data[f'arr_{i}']).float() for i in range(len(data.files))]

# Normalize all tensors
X = [Normalizer().normalize(tensor) for tensor in X]

# Load new features (assuming they are in another npz file or path)
fts_path = 'E:\\trans_effi\\data\\fts.npz'  # Replace with actual path for new features
with np.load(fts_path) as fts:
    X_fts = [torch.from_numpy(fts[f'arr_{i}']).float() for i in range(len(fts.files))]
X_fts = [Normalizer().normalize(tensor) for tensor in X_fts]

df_path = 'E:\\trans_effi\\data\\Transmission_Efficiency_Results.xlsx'
df = pd.read_excel(df_path)
df = df.drop(columns=df.columns[0])
y = torch.tensor(df.values).float().T


X = torch.stack(X).unsqueeze(1)
dataset = CustomDataset(X, X_fts, y)


total_count = len(dataset)
train_count = 39
val_count = 3
test_count = total_count - train_count - val_count
print(train_count, val_count, test_count)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_count, val_count, test_count])

generator = torch.Generator().manual_seed(316)


train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, generator = generator) #generator = generator
val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)

print("训练集样本索引:", train_dataset.indices)
print("验证集样本索引:", val_dataset.indices)
print("测试集样本索引:", test_dataset.indices)

set_seed(316)
# Model, Loss, Optimizer Setup
#model = Inception3DCNN()


# model = Light3DCNN()


#model = InceptionNet()

model = ExInceptionNet()


# model_params = {
#     'n_input_channels': 1,
#     'n_classes': 3,
#     'conv1_t_size': 3,
#     'conv1_t_stride': 1,
#     'no_max_pool': False,
#     'shortcut_type': 'B',
#     'widen_factor': 1.0
# }
#
# model = generate_model(34, **model_params)


ParameterCounter().count_parameters(model)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.004)
#optimizer = optim.SGD(model.parameters(), lr=0.002)#, momentum=0.9, weight_decay=5e-4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Activation Visualizer Setup
activation_visualizer = ActivationVisualizer(model)

# Training and Testing Functions

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=100, patience=5):
    model.train()
    training_losses, validation_losses = [], []
    early_stopping = EarlyStopping(save_path=f'E:\\trans_effi\\model_result\\trained_{type(model).__name__}.pth',
                                   patience=patience, verbose=True)
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        total_batches = 0

        # Training Loop
        with tqdm(train_loader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                x1, x2, labels = data
                x1,x2, labels = x1.to(device), x2.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = model(x1,x2)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_batches += 1
                tepoch.set_postfix(loss=running_loss / total_batches)

        # Calculate average training loss for the epoch
        average_train_loss = running_loss / total_batches
        training_losses.append(average_train_loss)
        print(f'Average Training Loss for Epoch {epoch + 1}: {average_train_loss:.4f}')

        # # Plot activation map
        # if (epoch + 1) % 20 == 0:
        #     activation = activation_visualizer.get_activation('conv1')
        #     activation_visualizer.plot_activation_maps(activation)
        #     activation = activation_visualizer.get_activation('conv2')
        #     activation_visualizer.plot_activation_maps(activation)
        #     activation = activation_visualizer.get_activation('conv3')
        #     activation_visualizer.plot_activation_maps(activation)

        # Evaluation Loop (Validation)
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        total_val_batches = 0

        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tval:
                for data in tval:
                    x1, x2, labels = data
                    x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                    outputs = model(x1,x2)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    total_val_batches += 1
                    tval.set_postfix(val_loss=total_val_loss / total_val_batches)

        # Calculate average validation loss for the epoch
        average_val_loss = total_val_loss / total_val_batches
        validation_losses.append(average_val_loss)
        print(f'Average Validation Loss for Epoch {epoch + 1}: {average_val_loss:.4f}')

        # Early stopping check
        early_stopping(average_val_loss,model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print('Finished Training')

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 5))
    # plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
    # plt.plot(range(1, len(validation_losses) + 1), validation_losses, color='red', label='Validation Loss')
    plt.plot(range(16, len(training_losses) + 1), training_losses[15:], label='Training Loss')
    plt.plot(range(16, len(validation_losses) + 1), validation_losses[15:], color='red', label='Validation Loss')
    plt.title('Training and Validation Loss Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate stability of the loss
    LossStabilityCalculator.calculate_stability(validation_losses, 'Validation')

    # Final Evaluation on Test Set
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0.0
    total_test_batches = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as ttest:
            for data in ttest:
                x1, x2, labels = data
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                outputs = model(x1,x2)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                total_test_batches += 1
                ttest.set_postfix(test_loss=total_test_loss / total_test_batches)

                # Store predictions and labels
                all_predictions.append(outputs.cpu())
                all_labels.append(labels.cpu())

    # Calculate average test loss
    average_test_loss = total_test_loss / total_test_batches
    print(f'Final Test Loss: {average_test_loss:.4f}')

    # Convert predictions and labels to tensors
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Print predictions and labels
    print("Predictions vs Labels:")
    for i in range(len(all_predictions)):
        print(f"Prediction: {all_predictions[i]}, Label: {all_labels[i]}")


    model_name = f'E:\\trans_effi\\model_result\\trained_{type(model).__name__}.pth'
    torch.save(model.state_dict(), model_name)
# Training the model
train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, patience=10)

