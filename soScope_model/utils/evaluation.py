import torch
import numpy as np
from utils.ReadData import get_dataset, get_spatial_dataset

# class Embedded_ST_Dataset:
#     def __init__(self, data_dir, num_neighbors, encoder, device='cpu'):
#         encoder = encoder.to(device)
#         self.data_dir = data_dir
#         self.num_neighbors = num_neighbors
#         self.reps = self._embed(encoder, device)
#
#     def _embed(self, encoder, device):
#         encoder.eval()
#         st_data = get_spatial_dataset(
#             self.data_dir,
#             self.num_neighbors)
#
#         for i, item in enumerate(st_data):
#             st_data[i] = item.to(device)
#
#         with torch.no_grad():
#             x, edge_index = st_data[:2]
#             p_z_given_x_a = self.encode(x, edge_index)
#             z = p_z_given_x_a.mean
#             reps = z.mean.detach()
#         return reps
#
#     def __getitem__(self, index):
#         y = self.reps[index]
#         return y
#
#     def __len__(self):
#         return self.reps.size()[0]
#
#
# class Embedded_Sub_Dataset:
#     def __init__(self, st_data_dir, num_neighbors, sub_data_dir,
#                  st_encoder, sub_encoder, device='cpu'):
#         st_encoder = st_encoder.to(device)
#         sub_encoder = sub_encoder.to(device)
#         self.st_data_dir = st_data_dir
#         self.sub_data_dir = sub_data_dir
#         self.num_neighbors = num_neighbors
#         self.reps = self._embed(st_encoder, sub_encoder, device)
#
#     def _embed(self, st_encoder, sub_encoder, device):
#         st_encoder.eval()
#         sub_encoder.eval()
#
#         st_data = get_spatial_dataset(
#             self.data_dir,
#             self.num_neighbors)
#
#         for i, item in enumerate(st_data):
#             st_data[i] = item.to(device)
#
#         with torch.no_grad():
#             x, edge_index = st_data[:2]
#             p_z_given_x_a = self.encode(x, edge_index)
#             z = p_z_given_x_a.mean
#             reps = z.mean.detach()
#         return reps
#
#     def __getitem__(self, index):
#         y = self.reps[index]
#         return y
#
#     def __len__(self):
#         return self.reps.size()[0]

def build_matrix(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    xs = []
    ys = []

    for x, y in data_loader:
        xs.append(x)
        ys.append(y)

    xs = torch.cat(xs, 0)
    ys = torch.cat(ys, 0)

    if xs.is_cuda:
        xs = xs.cpu()
    if ys.is_cuda:
        ys = ys.cpu()

    return xs.data.numpy(), ys.data.numpy()


def evaluate(encoder, train_on, test_on, device):
    embedded_train = EmbeddedDataset(train_on, encoder, device=device)
    embedded_test = EmbeddedDataset(test_on, encoder, device=device)
    return train_and_evaluate_linear_model(embedded_train, embedded_test)


def merge_evaluate(encoder_s, encoder_p, train_on, test_on, device):
    embedded_train = MergeEmbeddedDataset(train_on, encoder_s, encoder_p, device=device)
    embedded_test = MergeEmbeddedDataset(test_on, encoder_s, encoder_p, device=device)
    return train_and_evaluate_linear_model(embedded_train, embedded_test)


def train_and_evaluate_linear_model_from_matrices(x_train, y_train, solver='saga', multi_class='multinomial', tol=.1, C=10):
    model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
    model.fit(x_train, y_train)
    return model


def train_and_evaluate_linear_model(train_set, test_set, solver='saga', multi_class='multinomial', tol=.1, C=10):
    x_train, y_train = build_matrix(train_set)
    x_test, y_test = build_matrix(test_set)

    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
    model.fit(x_train, y_train)

    test_accuracy = model.score(x_test, y_test)
    train_accuracy = model.score(x_train, y_train)

    return train_accuracy, test_accuracy


def z_infer(st_data, encoder, device='cpu'):
    encoder.eval()
    for i, item in enumerate(st_data):
        st_data[i] = item.to(device)
    with torch.no_grad():
        x, edge_index = st_data[:2]
        p_z_given_x_a = encoder(x, edge_index)
        z = p_z_given_x_a.mean
        reps = z.detach()
    return reps
