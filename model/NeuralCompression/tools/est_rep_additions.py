import copy
import glob
import os
import re
import sys

import numpy as np
import tensorflow as tf
from graphviz import Graph

model_dir = '../trained/GTSRB48x48/0d2ce453e19aeb755f541df7d319b489724a3578_1'
DEBUG = True
sys.setrecursionlimit(2000)

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
    return weights_estimate, ss_error, binary_filter, a_new


def extract_step_nr(file):
    s = re.findall("\d+", file)[-1]
    return (int(s) if s else -1, file)


def get_shortest_dist_to_unused_node(current_node, binary_nodes):
    all_binary_filters = np.array([x.binary_filter for _, x in enumerate(binary_nodes)])
    current_node_filter = current_node.binary_filter
    dist = 1 - ((np.tensordot(all_binary_filters,
                              np.expand_dims(current_node_filter, axis=-1),
                              axes=([1, 2, 3], [0, 1, 2])) + np.prod(
        current_node_filter.shape)) / 2 / np.prod(
        current_node_filter.shape))
    # distance to it self would obviously be 0, thus one has to ignore the first result.
    # We do this by setting its distance to distance longer than 1 (more than 100% uncorrelated)
    dist[current_node.layer_filter_nr] = 1.01

    # We also need to make all linked nodes uninteresting
    dist[[x.layer_filter_nr for x in current_node.edges]] = 1.01
    dist[current_node.forbidden_edges] = 1.01
    return np.argmin(dist), np.min(
        dist)


def update_all_shortest_unused_node(binary_filter_nodes, remaining_nodes):
    if not remaining_nodes:
        return
    else:
        current_node = remaining_nodes.pop(0)
        update_single_shortest_node(binary_filter_nodes, current_node)
        update_all_shortest_unused_node(binary_filter_nodes, remaining_nodes)


def update_single_shortest_node(binary_filter_nodes, current_node):
    shortest_node, shortest_node_dist = get_shortest_dist_to_unused_node(current_node, binary_filter_nodes)
    current_node.closest_unused_edge = binary_filter_nodes[shortest_node]
    current_node.closest_unused_edge_dist = shortest_node_dist


def check_cycles(root, prev_root, remaining_children, visited_nodes):
    if root in visited_nodes:
        return True
    else:
        if not remaining_children:
            return False
        else:
            visited_nodes.append(root)
            if prev_root in remaining_children:
                remaining_children.remove(prev_root)
            while remaining_children:
                child = remaining_children.pop()
                if check_cycles(child, root, copy.copy(child.edges), visited_nodes):
                    return True
            return False


def traverse_single_graph(root, visited_nodes, graph=None):
    remaining_children = copy.copy(root.edges)
    visited_nodes.append(root)
    for x in visited_nodes:
        if x in remaining_children:
            remaining_children.remove(x)
    while remaining_children:
        child = remaining_children.pop()
        if graph:
            graph.edge(str(root.layer_filter_nr), str(child.layer_filter_nr), penwidth="0.1",
                       xlabel=str(child.layer_filter_nr), fontsize="1")

        traverse_single_graph(child, visited_nodes, graph)
    return visited_nodes


def mst(binary_nodes, remaining_nodes):
    if not remaining_nodes:
        return
    else:
        all_min_distance = [x.closest_unused_edge_dist for x in binary_nodes]
        min_distance_node = np.argmin(all_min_distance)
        cycle_test_binary_nodes = copy.deepcopy(binary_nodes)
        cycle_test_parent = cycle_test_binary_nodes[min_distance_node]
        cycle_test_parent.edges.append(cycle_test_parent.closest_unused_edge)
        cycle_test_new_child = cycle_test_parent.edges[-1]
        cycle_test_new_child.edges.append(cycle_test_parent)
        cycle_test_new_child.forbidden_edges.append(cycle_test_parent.layer_filter_nr)
        cycle_test_parent.forbidden_edges.append(cycle_test_new_child.layer_filter_nr)

        if not check_cycles(cycle_test_parent, [], copy.copy(cycle_test_parent.edges), []):
            del binary_nodes
            binary_nodes = cycle_test_binary_nodes
            nodes_in_tree = 0
            for x in traverse_single_graph(cycle_test_parent, []):
                if x in binary_nodes:
                    nodes_in_tree += 1

            if DEBUG:
                if nodes_in_tree >= len(binary_nodes):
                    g = Graph('G', filename='process.gv', engine='sfdp',
                              node_attr={'shape': 'point', 'width': '0.005', 'fixedsize': 'true', 'fontsize': '0'})
                    traverse_single_graph(cycle_test_parent, [], g)
                    g.view()
            if nodes_in_tree >= len(binary_nodes):
                return

        else:
            binary_nodes[min_distance_node].forbidden_edges.append(cycle_test_new_child.layer_filter_nr)
            binary_nodes[cycle_test_new_child.layer_filter_nr].forbidden_edges.append(cycle_test_parent.layer_filter_nr)
            del cycle_test_binary_nodes

        update_single_shortest_node(binary_nodes, binary_nodes[min_distance_node])
        update_single_shortest_node(binary_nodes, binary_nodes[cycle_test_new_child.layer_filter_nr])
        mst(binary_nodes, remaining_nodes)


def main():
    list_of_graphs = glob.glob('{}/*.meta'.format(model_dir))
    last_graph = max(list_of_graphs, key=extract_step_nr)
    saver = tf.train.import_meta_graph(last_graph)
    with tf.Session() as sess:
        saver.restore(sess, os.path.splitext(last_graph)[0])
        conv_kernel_tensor = tf.trainable_variables(
            scope='conv2d_1')  # first tensor are kernel weights in this model
        conv_weights = conv_kernel_tensor[0].eval()
        conv_weights_approx = np.copy(conv_weights)
        binary_filter_layer = np.ndarray(np.append(np.shape(conv_weights)[:-1], [6, np.shape(conv_weights)[-1]]))

        # calculate all binary filters and stack them together into a single array for a layer
        for i in range(np.shape(conv_weights)[-1]):
            weights_3d = conv_weights[:, :, :, i]
            weights_flat = np.ndarray.flatten(weights_3d)
            approx_flat, ss_error, binary_filter, alpha = m_fold_binary_approx(weights=weights_flat,
                                                                               num_binary_filter=3,
                                                                               max_num_binary_filter=6,
                                                                               max_opt_runs=1000)
            conv_weights_approx[:, :, :, i] = np.reshape(approx_flat, np.shape(conv_weights)[:-1])
            binary_filter = np.reshape(binary_filter, np.append(np.shape(conv_weights)[:-1], 6))
            binary_filter_layer[:, :, :, :, i] = binary_filter

        # calculate the shortest path for each filter
        binary_filter_nodes = []

        for i in range(binary_filter_layer[:, :, :, 5, :].shape[-1]):
            x = binary_filter_layer[:, :, :, 5, i]
            binary_filter_nodes.append(
                BinaryFilterNode(binary_filter=x, binary_filter_nr=5, layer_filter_nr=i, edges=[],
                                 closest_unused_edge=[]))

        remaining_nodes = copy.copy(binary_filter_nodes)
        update_all_shortest_unused_node(binary_filter_nodes, remaining_nodes)

        # # test implementation
        # binary_filter_nodes[0].edges.append(binary_filter_nodes[1])
        # binary_filter_nodes[1].edges.append(binary_filter_nodes[0])
        #
        # binary_filter_nodes[1].edges.append(binary_filter_nodes[2])
        # binary_filter_nodes[2].edges.append(binary_filter_nodes[1])
        #
        # binary_filter_nodes[3].edges.append(binary_filter_nodes[4])
        # binary_filter_nodes[4].edges.append(binary_filter_nodes[3])
        #
        #
        # binary_filter_nodes[4].edges.append(binary_filter_nodes[1])
        # binary_filter_nodes[1].edges.append(binary_filter_nodes[4])
        # cyclic = check_cycles(binary_filter_nodes[0], [], copy.copy(binary_filter_nodes[0].edges), [])

        remaining_nodes = copy.copy(binary_filter_nodes)
        mst(binary_filter_nodes, remaining_nodes)
        binary_filter_nodes


class BinaryFilterNode:

    def __init__(self, binary_filter, binary_filter_nr, layer_filter_nr, edges=None,
                 closest_unused_edge=None, closest_unused_edge_dist=None):
        self.binary_filter = binary_filter
        self.binary_filter_nr = binary_filter_nr
        self.layer_filter_nr = layer_filter_nr
        self.edges = edges
        self.closest_unused_edge = closest_unused_edge
        self.closest_unused_edge_dist = closest_unused_edge_dist
        self.forbidden_edges = []
        self.__marked = False

    def set_marked(self):
        self.__marked = True

    def unset_marked(self):
        self.__marked = False


if __name__ == '__main__':
    main()
