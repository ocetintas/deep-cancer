import numpy as np
import torch

from MDN_conv3d import MDN
from BatchLoader_conv3d import BatchLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

import os
import time
import statistics

# -------------------------------------------------------
# PARAMETERS
device = "cpu"
#device = "cuda:0"
EPOCH = 50
NUM_GAUSSIAN = 2

BATCH_SIZE = 8
LR = 0.0003
# -------------------------------------------------------


# Save training and validation error
def progress(batch_loader, net):
    with torch.no_grad():

        # Loss containers
        train_list = []
        val_list = []

        # Train Loss
        batch_loader.create_train_batches()
        for b in batch_loader.mini_batches[:int(len(batch_loader.mini_batches)/5)]:
            X, Y = batch_loader.get_XY(b)
            logits, means, precisions, sumlogdiag = net.forward(X)
            train_loss = net.loss(Y, logits, means, precisions, sumlogdiag)
            train_list.append(train_loss.item())

        # Validation Loss
        for b in batch_loader.val_batches:
            X, Y = batch_loader.get_XY(b)
            logits, means, precisions, sumlogdiag = net.forward(X)
            val_loss = net.loss(Y, logits, means, precisions, sumlogdiag)
            val_list.append(val_loss.item())

        print(train_list)
        print(val_list)
        train_loss = statistics.mean(train_list)
        val_loss = statistics.mean(val_list)
        print("\n")
        print("-------------------")
        print("Training loss: ", train_loss)
        print("Validation loss: ", val_loss)
        print("-------------------")

        return train_loss, val_loss


# Create network, loader and optimizer
network = MDN(num_gaussian=NUM_GAUSSIAN, num_output_features=7, device=device).to(device)
loader = BatchLoader(device=device, batch_size=BATCH_SIZE)
optimizer = torch.optim.Adam(network.parameters(), lr=LR)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

# Training and validation loss history containers
training_loss_history = []
validation_loss_history = []

# Handles loading for performance measurement - Validation, Test etc.
performance_loader = BatchLoader(device=device, batch_size=8)
performance_loader.create_testval_batches()

# Print and save the values at the initialization
# t_loss, v_loss = progress(performance_loader, network)
# training_loss_history.append(t_loss)
# validation_loss_history.append(v_loss)

for epoch in tqdm(range(EPOCH)):

    loader.create_train_batches()

    for mini_batch in loader.mini_batches:
        X, Y = loader.get_XY(mini_batch)    # X and Y are already in the device

        logits, means, precisions, sumlogdiag = network.forward(X)
        loss = network.loss(Y, logits, means, precisions, sumlogdiag)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

    # Save the losses
    t_loss, v_loss = progress(performance_loader, network)
    training_loss_history.append(t_loss)
    validation_loss_history.append(v_loss)

    # Learning rate decay
    lr_scheduler.step()


# Visualize 10 examples and their results
with torch.no_grad():
    loader = BatchLoader(device=device, batch_size=2)
    loader.create_testval_batches()
    for val_ex in loader.val_batches[:8]:
        X, Y = loader.get_XY(val_ex)

        logits, means, precisions, sumlogdiag = network.forward(X)
        loss = network.loss(Y, logits, means, precisions, sumlogdiag)

        print("--------------")
        print(loss)
        print(Y)
        print(logits)
        print(means)

        print(sumlogdiag)
        print("--------------")


# Save the model - Both Model and Dictionary Version
time_stamp = str(time.time())   # Ensure different name for each model to not overwrite
path = os.path.join("models", time_stamp)
os.mkdir(path)
model_name = os.path.join(path, time_stamp + "_model")
state_dict_name = os.path.join(path, time_stamp + "_state")

parameters = {"epoch": EPOCH, "lr": LR, "batch_size": BATCH_SIZE}
parameters_name = os.path.join(path, time_stamp + "_parameters.pkl")

loss_history = {"train": training_loss_history, "validation": validation_loss_history}
loss_name = os.path.join(path, "loss_history.pkl")

# Save the model
torch.save(network, model_name)
torch.save(network.state_dict(), state_dict_name)
# torch.save(parameters, parameters_name)

# Save parameters and loss history
with open(parameters_name, 'wb') as handle:
    pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(loss_name, 'wb') as handle:
    pickle.dump(loss_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Visualize the training curves
plt.plot(training_loss_history, "b")
plt.plot(validation_loss_history, "g")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")

# Save the figure
fig_name = os.path.join(path, "training_loss.png")
plt.savefig(fig_name)

# Display the image
plt.show()
