import numpy as np

def pred(w, x):
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
    return np.array_equal(pred(w, X), y)

def perceptron(X, y, w_init, max_iter=1000):
    w = [w_init]
    d = X.shape[0]
    iter_count = 0

    mis_points = []
    while iter_count < max_iter:
        # Trộn dữ liệu
        mix_id = np.random.permutation(X.shape[1])
        converged = True
        for i in range(X.shape[1]):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if pred(w[-1], xi)[0] != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi * xi
                w.append(w_new)
                converged = False  # Nếu có lỗi phân loại, chưa hội tụ

        iter_count += 1
        if converged:
            break

    return (w, mis_points)

if __name__ == '__main__':
    # Đọc dữ liệu từ file
    data = np.genfromtxt('dada.csv', delimiter=',', skip_header=1)

    # Loại bỏ dòng chứa dữ liệu không xác định
    data = data[~np.isnan(data).any(axis=1)]

    # Tách features và nhãn
    X = data[:, 1:-1].T
    y = data[:, -1].reshape(1, -1)

    # Chuyển đổi nhãn lớp thành số nguyên
    y = y.astype(int)

    # Thêm bias
    Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

    # Khởi tạo trọng số
    w_init = np.random.randn(Xbar.shape[0], 1)

    # Huấn luyện perceptron
    (w, m) = perceptron(Xbar, y, w_init)

    # In ra trọng số
    print("Trọng số cuối cùng:", w[-1].T)