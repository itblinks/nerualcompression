import numpy as np
import matplotlib.pyplot as plt


def m_fold_binaryconv_tensor_approx_whole(weights, num_binary_filter, paper_approach=False, use_pow_two=False):
    weights_estimate = np.zeros_like(weights)
    ss_error = 0
    for i in range(np.shape(weights)[-1]):
        weights_3d = weights[:, :, :, i]
        weights_flat = np.ndarray.flatten(weights_3d)
        weights_estimate_flat, error = m_fold_binary_approx(weights_flat, num_binary_filter, paper_approach,
                                                            use_pow_two)
        weights_estimate[:, :, :, i] = np.reshape(weights_estimate_flat, np.shape(weights_3d))
        ss_error += error
    return weights_estimate, ss_error


def m_fold_binary_tensor_approx_channel(weights, num_binary_filter, paper_approach=False):
    return


def m_fold_binarydense_tensor_approx_channel(weights, num_binary_filter, paper_approach=False, use_pow_two=False):
    weights_estimate = np.zeros_like(weights)
    ss_error = 0
    for i in range(np.shape(weights)[-1]):
        weights_neuron = weights[:, i]
        weights_estimate_channel, error = m_fold_binary_approx(weights_neuron, num_binary_filter, paper_approach,
                                                               use_pow_two)
        weights_estimate[:, i] = weights_estimate_channel
        ss_error += error
    return weights_estimate, ss_error


def m_fold_binary_approx(weights, num_binary_filter, paper_approach=False, use_pow_two=False):
    M_max = 3
    RUN_MAX = 1000
    binary_filter = np.zeros((np.shape(weights)[0], M_max))
    binary_filter_old = np.ones_like(weights)
    weights_new = np.copy(weights)

    if paper_approach:
        for i in range(num_binary_filter):
            u_i = -1 + i * 2 / (num_binary_filter - 1)
            binary_filter[:, i] = np.sign(weights_new - np.mean(weights_new) + u_i * np.std(weights_new))
    else:
        for i in range(M_max):
            binary_filter[:, i] = ((weights_new >= 0).astype(float) * 2 - 1) * binary_filter_old
            weights_new = np.abs(weights_new)
            weights_new = weights_new - np.mean(weights_new)
            binary_filter_old = binary_filter[:, i]

        binary_filter_opt = np.copy(binary_filter)
        stop_cond = False
        runs = 1
        while not stop_cond:
            binary_filter = np.copy(binary_filter_opt)
            a_new = np.linalg.lstsq(binary_filter, weights, rcond=None)[0]
            binary_filter_old = np.ones_like(weights)
            weights_new = np.copy(weights)
            for i in range(M_max):
                binary_filter_opt[:, i] = ((weights_new >= 0).astype(float) * 2 - 1) * binary_filter_old
                weights_new = np.abs(weights_new)
                weights_new = weights_new - a_new[i]
                binary_filter_old = binary_filter_opt[:, i]
            if np.array_equal(binary_filter, binary_filter_opt) or runs >= RUN_MAX:
                stop_cond = True
            else:
                runs += 1

    if use_pow_two:
        shift = np.floor(np.log(np.abs(a_new)*4.0/3.0) / np.log(2.0))
        a_new = np.sign(a_new) * np.power(2.0, shift)

    weights_estimate = np.zeros_like(weights)
    for i in range(num_binary_filter):
        weights_estimate = weights_estimate + a_new[i] * binary_filter[:, i]

    ss_error = np.sum(np.power(weights_estimate - weights, 2))
    return weights_estimate, ss_error


def log_approx(W, result_plots=False):
    W_reshape = np.reshape(W, newshape=[1, -1])
    shift = np.floor(np.log(np.abs(W_reshape)*4.0/3.0) / np.log(2.0))
    W_est = np.sign(W_reshape) * np.power(2.0, shift)
    # illustrative plots
    if result_plots:
        plt.hist(np.transpose(W_est),
                 bins=[0, 0.01, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.12, 0.13, 0.245, 0.255, 0.495, 0.505, 0.995,
                       1.005],
                 histtype='barstacked')
        plt.show()
    W_est = W_est.reshape(W.shape)
    SSerror = np.sum(np.power(W_est - W, 2))
    return W_est, SSerror

if __name__ == '__main__':
    m_fold_binary_approx(np.random.rand(1000), 3)
