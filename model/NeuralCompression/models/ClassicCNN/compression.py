import numpy as np
import matplotlib.pyplot as plt


def m_fold_binaryconv_tensor_approx_whole(weights, num_binary_filter, max_num_binary_filter, paper_approach=False,
                                          use_pow_two=False, max_opt_runs=1000):
    weights_estimate = np.zeros_like(weights)
    ss_error = 0
    for i in range(np.shape(weights)[-1]):
        weights_3d = weights[:, :, :, i]
        weights_flat = np.ndarray.flatten(weights_3d)
        weights_estimate_flat, error = m_fold_binary_approx(weights=weights_flat, num_binary_filter=num_binary_filter,
                                                            paper_approach=paper_approach,
                                                            use_pow_two=use_pow_two,
                                                            max_num_binary_filter=max_num_binary_filter,
                                                            max_opt_runs=max_opt_runs)

        weights_estimate[:, :, :, i] = np.reshape(weights_estimate_flat, np.shape(weights_3d))
        ss_error += error
    return weights_estimate, ss_error


def m_fold_binarydense_tensor_approx_channel(weights, num_binary_filter, max_num_binary_filter, paper_approach=False,
                                             use_pow_two=False, max_opt_runs=1000):
    weights_estimate = np.zeros_like(weights)
    ss_error = 0
    for i in range(np.shape(weights)[-1]):
        weights_neuron = weights[:, i]
        weights_estimate_channel, error = m_fold_binary_approx(weights=weights_neuron,
                                                               num_binary_filter=num_binary_filter,
                                                               paper_approach=paper_approach,
                                                               use_pow_two=use_pow_two,
                                                               max_num_binary_filter=max_num_binary_filter,
                                                               max_opt_runs=max_opt_runs)
        weights_estimate[:, i] = weights_estimate_channel
        ss_error += error
    return weights_estimate, ss_error


def m_fold_binary_approx(weights, num_binary_filter, max_num_binary_filter, paper_approach=False, use_pow_two=False,
                         max_opt_runs=1000):
    """
    Perform M fold binary approximation.

    This function performs a M fold approximation and calculates a weight estimate when using binary approximation. The
    estimated weigth can then be loaded by the model to estimate the effect of the binary approximation.

    The function calculates 'max_num_binary_filter' binary filters $B$ containing (1,-1) and real valued stretching
    factors $alpha$. Afterwards an approximation is calculated by

    $W_approx = a_1*B_1 + a_2*B_2 + ... + a_{num_binary_filter}*B_{num_binary_filter}$

    This means that from the total of 'max_num_binary_filter', only num_binary_filter are going to be used in the
    approximation

    Parameters
    ----------
    weights : ndarray
        1D vector cntaining all the weights which need to be approximated
    num_binary_filter : int
        Integer representing the number of binary filters which are going to be used for the
        approximation
    max_num_binary_filter : int
        The number of calculated binary filters
    paper_approach : bool
        OPTIONAL: Use the approach described by Guo et al. in network sketching.
        Defaults to False
    use_pow_two : bool
        OPTIONAL: Round the $\alpha$ values to the next power of two. Defaults to False
    max_opt_runs : int
        OTIONAL: Maximum number of optimization runs for the calculation of the \alpha's
        Defaults to 1000

    Returns
    -------
    tuple
        tuple containing the approximated weights and the squared error compared to the original filter

    """

    binary_filter = np.zeros((np.shape(weights)[0], max_num_binary_filter))
    binary_filter_old = np.ones_like(weights)
    weights_new = np.copy(weights)

    if paper_approach:
        a_new = np.zeros([max_num_binary_filter])
        for i in range(max_num_binary_filter):
            binary_filter[:, i] = ((weights_new >= 0).astype(float) * 2 - 1)
            a_new[i] = np.linalg.lstsq(binary_filter, weights, rcond=None)[0][i]
            weights_new = weights_new - a_new[i] * binary_filter[:, i]

    else:
        for i in range(max_num_binary_filter):
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
            for i in range(max_num_binary_filter):
                binary_filter_opt[:, i] = ((weights_new >= 0).astype(float) * 2 - 1) * binary_filter_old
                weights_new = np.abs(weights_new)
                weights_new = weights_new - a_new[i]
                binary_filter_old = binary_filter_opt[:, i]
            if np.array_equal(binary_filter, binary_filter_opt) or runs >= max_opt_runs:
                stop_cond = True
            else:
                runs += 1

    if use_pow_two:
        shift = np.floor(np.log(np.abs(a_new) * 4.0 / 3.0) / np.log(2.0))
        a_new = np.sign(a_new) * np.power(2.0, shift)

    weights_estimate = np.zeros_like(weights)
    for i in range(num_binary_filter):
        weights_estimate = weights_estimate + a_new[i] * binary_filter[:, i]

    ss_error = np.sum(np.power(weights_estimate - weights, 2))
    return weights_estimate, ss_error


def log_approx(W, result_plots=False):
    W_reshape = np.reshape(W, newshape=[1, -1])
    shift = np.floor(np.log(np.abs(W_reshape) * 4.0 / 3.0) / np.log(2.0))
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
    np.random.seed(1)
    m_fold_binary_approx(np.random.rand(1000), max_num_binary_filter=6, num_binary_filter=3, max_opt_runs=1000,
                         paper_approach=False)
