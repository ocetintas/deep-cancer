import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from BatchLoader_conv3d import BatchLoader


class MDN(nn.Module):
    """
    Modified version of MDN_conv3d. Additional features:
        - Xavier Initialization
        - Batch Normalization
        - Residual Blocks
    """

    def __init__(self, num_gaussian, num_output_features=7, device="cpu"):

        super().__init__()
        self.num_gaussian = num_gaussian    # Number of mixtures to be used
        self.num_output_features = num_output_features  # Dx, rho, icx, icy, icz, uth, T (Total 7)

        self.num_upper_params = (num_output_features * (num_output_features - 1)) // 2
        self.row_ix, self.column_ix = np.triu_indices(num_output_features, k=1)
        self.diag_ix = range(num_output_features)
        self.epsilon = 1e-2
        self.device = device

        # ARCHITECTURE
        # Convolution - 1
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_normal_(self.conv1.weight, gain=2)
        self.bn1 = nn.BatchNorm3d(8)
        self.act1 = nn.ReLU()
        self.mpool1 = nn.MaxPool3d(kernel_size=2)
        self.drop1 = nn.Dropout3d(0.1)

        # Convolution - 2
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_normal_(self.conv2.weight, gain=2)
        self.bn2 = nn.BatchNorm3d(16)
        self.act2 = nn.ReLU()
        self.mpool2 = nn.MaxPool3d(kernel_size=2)
        self.drop2 = nn.Dropout3d(0.1)

        # Convolution - 3
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_normal_(self.conv3.weight, gain=2)
        self.bn3 = nn.BatchNorm3d(32)
        self.act3 = nn.ReLU()
        self.mpool3 = nn.MaxPool3d(kernel_size=2)
        self.drop3 = nn.Dropout3d(0.1)

        # Convolution - 4
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_normal_(self.conv4.weight, gain=2)
        self.bn4 = nn.BatchNorm3d(64)
        self.act4 = nn.ReLU()
        self.mpool4 = nn.MaxPool3d(kernel_size=2)
        self.drop4 = nn.Dropout3d(0.1)

        # Fully Connected - 1
        self.fc1 = nn.Linear(2**15, 256)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc1_act = nn.Tanh()
        self.fc1_drop = nn.Dropout(0.1)

        # Gaussian parameters
        self.pi_layer = nn.Linear(256, num_gaussian)
        nn.init.xavier_normal_(self.pi_layer.weight)

        self.mean_layer = nn.Linear(256, num_gaussian * num_output_features)
        nn.init.xavier_normal_(self.mean_layer.weight)

        self.diagonal_layer = nn.Linear(256, num_gaussian * num_output_features)
        nn.init.xavier_normal_(self.diagonal_layer.weight)

        self.upper_layer = nn.Linear(256, num_gaussian * self.num_upper_params)
        nn.init.xavier_normal_(self.upper_layer.weight)

        # Resblock Downsample with 1x1 convolution
        self.res1 = nn.Conv3d(8, 32, kernel_size=1, stride=2)

    def forward(self, X):
        """
        N: Number of data in a minibatch
        M: Number of Gaussians
        C: Number of output parameters

        :param X: torch.tensor containing training data | X.shape = (N, 129*129*129)
        :return:
        logits: pi before normalization | logits.shape = (N, M)
        means: means.shape = (N, M, C)
        precisions: precisions.shape = (N, M, C, C)
        sumlogdiag: sumlogdiad.shape = (N, M)
        """

        # Convolution - 1
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.act1(X)
        X = self.mpool1(X)
        X = self.drop1(X)

        # Convolution - 2
        residual = X    # Will be used for residual block
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.act2(X)
        X = self.mpool2(X)
        X = self.drop2(X)

        # Convolution - 3
        X = self.conv3(X)
        X = self.bn3(X)
        residual = self.res1(residual)  # Match the shapes
        X += residual
        X = self.act3(X)
        X = self.mpool3(X)
        X = self.drop3(X)

        # Convolution - 4
        X = self.conv4(X)
        X = self.bn4(X)
        X = self.act4(X)
        X = self.mpool4(X)
        X = self.drop4(X)

        # Flatten Layer
        X = X.view(X.shape[0], -1)

        # Fully connected layer
        X = self.fc1(X)
        X = self.fc1_bn(X)
        X = self.fc1_act(X)
        h = self.fc1_drop(X)

        # pi and mean
        logits = self.pi_layer(h)     # shape = (N, M)
        means = self.mean_layer(h).view(-1, self.num_gaussian, self.num_output_features)    # shape = (N, M, C)

        # sumlogdiag and precisions
        unconstrained_diagonal = self.diagonal_layer(h).view(-1, self.num_gaussian, self.num_output_features)
        upper = self.upper_layer(h).view(-1, self.num_gaussian, self.num_upper_params)
        diagonal = F.softplus(unconstrained_diagonal) + self.epsilon

        precision_factors = torch.zeros(means.shape[0], self.num_gaussian, self.num_output_features,
                                        self.num_output_features).to(self.device)
        precision_factors[..., self.diag_ix, self.diag_ix] = diagonal
        precision_factors[..., self.row_ix, self.column_ix] = upper

        precisions = torch.matmul(torch.transpose(precision_factors, 2, 3), precision_factors)
        sumlogdiag = torch.sum(torch.log(diagonal), dim=-1)

        return logits, means, precisions, sumlogdiag

    def loss(self, Y, logits, means, precisions, sumlogdiag):
        """
        :param Y: torch.tensor containing the ground truths | Y.shape = (N, C)
        :return: torch.tensor containing the average loss of the minibatch
        """

        batch_size, n_mixtures, output_dim = means.size()
        Y = Y.view(-1, 1, output_dim)

        # Split the calculation
        a = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        b = -(output_dim / 2.0) * np.log(2 * np.pi)
        c = sumlogdiag
        d1 = (Y.expand_as(means) - means).view(batch_size, n_mixtures, output_dim, 1)
        d2 = torch.matmul(precisions, d1)
        d = -0.5 * torch.matmul(torch.transpose(d1, 2, 3), d2).view(
            batch_size, n_mixtures
        )

        # Mini batch loss
        mini_batch_loss = -torch.logsumexp(a + b + c + d, dim=-1)

        return torch.mean(mini_batch_loss, dim=0)


if __name__ == "__main__":
    loader = BatchLoader(device="cpu", batch_size=8)
    loader.create_testval_batches()
    X, Y = loader.get_XY(loader.test_batches[0])
    net = MDN(num_gaussian=2)
    net.forward(X)
