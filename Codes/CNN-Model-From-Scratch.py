import numpy as np


def zero_padding(X, pad):
    # Pad with zeros all images of the dataset X
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))
    return X_pad


def convolution_single_step(a_slice_prev, W, b):
    # Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    # of the previous layer.
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)
    return Z


def convolution_forward(A_prev, W, b, hparameters):
    # Implements the forward propagation for a convolution function
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_padding(A_prev, pad)
    for i in range(m):  # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i, :, :, :]
        for h in range(n_H):
            vert_start = h * stride
            vert_end = h * stride + f
            for w in range(n_W):  # loop over horizontal axis of the output volume
                horiz_start = w * stride
                horiz_end = w * stride + f
                for c in range(n_C):  # loop over channels (= #filters) of the output volume
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = convolution_single_step(a_slice_prev, weights, biases)
    cache = (A_prev, W, b, hparameters)
    return Z, cache


def pooling_forward(A_prev, hparameters, mode="max"):
    # Implements the forward pass of the pooling layer
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):  # loop over the training examples
        for h in range(n_H):  # loop on the vertical axis of the output volume
            vert_start = h * stride
            vert_end = h * stride + f
            for w in range(n_W):  # loop on the horizontal axis of the output volume
                horiz_start = w * stride
                horiz_end = w * stride + f
                for c in range(n_C):
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    cache = (A_prev, hparameters)
    return A, cache

