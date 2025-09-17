import numpy as np
from scipy import sparse

# Tạo dữ liệu

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
C = 3   # số lớp

# Sinh dữ liệu Gaussian cho 3 lớp
x0 = np.random.multivariate_normal(means[0], cov, N)
x1 = np.random.multivariate_normal(means[1], cov, N)
x2 = np.random.multivariate_normal(means[2], cov, N)

# Gộp dữ liệu
X = np.concatenate((x0, x1, x2), axis=0).T  # shape (2, 1500)
X = np.concatenate((np.ones((1, 3*N)), X), axis=0)  # thêm bias → shape (3, 1500)

# Nhãn gốc
original_label = np.asarray([0]*N + [1]*N + [2]*N)


# One-hot encoding

def convert_labels(y, C=C):
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y

Y = convert_labels(original_label, C)


# Hàm softmax

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # tránh tràn số
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


# Hàm loss (cross-entropy)

def cost(X, Y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y * np.log(A + 1e-12))  # thêm epsilon tránh log(0)


# Gradient

def grad(X, Y, W):
    A = softmax(W.T.dot(X))
    E = A - Y
    return X.dot(E.T)

# =====================
# Kiểm tra gradient (numerical vs analytical)
# =====================
def numerical_grad(X, Y, W, cost):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps
            W_n[i, j] -= eps
            g[i, j] = (cost(X, Y, W_p) - cost(X, Y, W_n)) / (2 * eps)
    return g


# Thuật toán softmax regression (SGD)

def softmax_regression(X, y, W_init, eta, tol=1e-4, max_count=10000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]

    count = 0
    check_w_after = 20
    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta * xi.dot((yi - ai).T)
            count += 1
            if count % check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
    return W


# Chạy thử

d = X.shape[0]
W_init = np.random.randn(d, C)
eta = 0.05

W = softmax_regression(X, original_label, W_init, eta)
print("Trọng số cuối cùng:\n", W[-1])
