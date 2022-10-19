import numpy as np


class Node:
    def __init__(self, attribute=None, value=None, left=None, right=None, leaf=False):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf


def find_split(data):
    # Finds the attribute and value with the highest information gain
    # Sort columns -> take median -> for each column, determine information gain with median
    max_gain = 0
    best_attribute, best_value = 0, 0
    best_l_split, best_r_split = [[]], [[]]
    i = 0
    for col in data[:, :-1].T:
        sorted_col = np.sort(col)
        median = np.median(np.sort(col))
        l_split = data[col < median]  # Half to left of median
        r_split = data[col >= median]  # Half to right of median
        curr_gain = info_gain(data, l_split, r_split)

        if curr_gain > max_gain:
            max_gain = curr_gain
            best_l_split = l_split
            best_r_split = r_split
            best_attribute = i
            best_value = median
        i += 1
    return best_attribute, best_value, best_l_split, best_r_split


def info_gain(dataset, left, right):
    # dataset is whole dataset, left is left part of where splitting, same for right
    h_all = entropy(dataset)
    left_size = left.size
    right_size = right.size
    total_size = left_size + right_size
    remainder = (left_size / total_size) * entropy(left) + (right_size / total_size) * entropy(right)
    gain = h_all - remainder
    return gain


def entropy(dataset):
    # Calculate entropy of input set
    rooms = dataset[:, -1]
    unique, counts = np.unique(rooms, return_counts=True)
    total = counts.sum()
    probs = counts / total
    hs = probs * np.log2(probs) * -1
    return hs.sum()


def decision_tree_learning(training_dataset, depth) -> (Node, int):
    y = training_dataset[:, -1]

    if np.min(y) == np.max(y):  # All samples have same categorical value
        return Node(value=y[0], leaf=True), depth

    attr, val, l_dataset, r_dataset = find_split(training_dataset)
    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
    node = Node(attribute=attr, value=val, left=l_branch, right=r_branch)

    return node, max(l_depth, r_depth)


def fit(model, sample):
    if model.leaf:
        return model.value
    else:
        if sample[model.attribute] < model.value:
            return fit(model.left, sample)
        else:
            return fit(model.right, sample)


if __name__ == "__main__":
    data_clean = np.loadtxt("./wifi_db/clean_dataset.txt")
    root_clean, depth_clean = decision_tree_learning(data_clean, 0)
    print(fit(root_clean, [-50, -68, -48, -56, -63, -80, -77]))
    # data_noisy = np.loadtxt("./wifi_db/noisy_dataset.txt")
    # print(data_noisy)
