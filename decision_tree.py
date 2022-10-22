import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt


class Node:
    def __init__(self, attribute=None, value=None, left=None, right=None, leaf=False):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.leaf = leaf

    def __str__(self):
        return f"Node - Left: [{self.left}] - Right[{self.right}]"


def find_split(data):
    # Finds the attribute and value with the highest information gain
    # Sort a column -> take two adjacent values mean -> calculate info gain on this mean
    max_gain = 0
    best_attribute, best_value = 0, 0
    best_l_split, best_r_split = [[]], [[]]
    attr = 0
    h_all = entropy(data)
    for col in data[:, :-1].T:
        sorted_col = np.sort(col)
        for i in range(sorted_col.size - 1):
            mp = (sorted_col[i] + sorted_col[i + 1]) / 2  # Mean of adjacent values
            l_split = data[col < mp]  # Data to left of midpoint
            r_split = data[col >= mp]  # Data to right of midpoint
            curr_gain = info_gain(h_all, l_split, r_split)

            if curr_gain > max_gain:
                max_gain = curr_gain
                best_l_split = l_split
                best_r_split = r_split
                best_attribute = attr
                best_value = mp
        attr += 1
    return best_attribute, best_value, best_l_split, best_r_split


def info_gain(h_all, left, right):
    # left is the left part of split, right is the right part
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


def decision_tree_learning(training_dataset, depth):
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
        return int(model.value)
    if sample[model.attribute] < model.value:
        return fit(model.left, sample)
    return fit(model.right, sample)


def split_dataset(x, test_proportion, random_generator=default_rng()):
    size = len(x)
    shuffled_indices = random_generator.permutation(size)
    n_train = round(size * (1 - test_proportion))
    x_train = x[shuffled_indices[:n_train]]
    x_test = x[shuffled_indices[n_train:]]
    return x_train, x_test


def evaluate(model, test_dataset):
    success = 0
    for sample in test_dataset:
        result = fit(model, sample[:-1])
        if result == sample[-1]:
            success += 1
    accuracy = 100 * (success / test_dataset.shape[0])
    return accuracy


def update_confusion_matrix(confusion_matrix, model, test):
    for sample in test:
        x, y = sample[:-1], int(sample[-1])
        result = fit(model, x)
        # x-axis predicted labels, y-axis actual labels
        confusion_matrix[(y - 1, result - 1)] += 1
    return confusion_matrix


def calc_recall_precision(confusion_matrix):
    # Calculate recall and precision rates and f1 measures per class (as percentages)
    precision = []
    recall = []
    f1 = []

    for i in range(4):
        tp = confusion_matrix[i][i]
        tp_scaled = 100 * tp  # for percentages
        precision.append(tp_scaled / np.sum(confusion_matrix[:, i], axis=0))
        recall.append(tp_scaled / np.sum(confusion_matrix[i]))
        f1.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))

    return recall, precision, f1


def prune(model, validation_set, root):
    #  Save some data for validation set
    #  After creating model, prune:
    #  Find node connected to two leaves ->
    #  Replace node with majority class label ->
    #  Compare accuracies
    if model.leaf:
        return
    print(model.attribute)
    print(validation_set[:, model.attribute])
    l_split = np.where(validation_set[:, model.attribute] < model.value)
    r_split = np.where(validation_set[:, model.attribute] >= model.value)

    if model.left.leaf and model.right.leaf:
        accuracy_before = evaluate(root, validation_set)
        values, counts = np.unique(validation_set[:, -1], return_counts=True)
        majority = values[list(counts).index(max(counts))]
        l_old, r_old, a_old = model.left, model.right, model.attribute
        model.attribute = majority
        model.left, model.right, model.leaf = None, None, True
        accuracy_after = evaluate(root, validation_set)
        if accuracy_after <= accuracy_before:  # Lower accuracy, revert
            model.left, model.right, model.attribute = l_old, r_old, a_old
        return

    prune(model.right, r_split, root)
    prune(model.left, l_split, root)


def create_pruned_tree(data):
    print("Creating pruned tree")
    data, validation = split_dataset(data, 0.1)
    train, test = split_dataset(data, 0.1)
    model, depth = decision_tree_learning(train, 0)
    print(f"Depth {depth}")
    print("Model created...")
    print("Pre-pruned evaluation:")
    print(evaluate(model, test))
    pruned_model = model
    prune(pruned_model, validation, model)
    while pruned_model != model:
        print("Pruning...")
        model = pruned_model
        prune(pruned_model, validation, model)
    print("Pruned evaluation:")
    print(evaluate(pruned_model, test))
    return pruned_model


def k_fold_cross_validation(dataset, k):
    # Split data k equal ways
    # Repeat k times:
    #   Take a subset of data to test
    #   Train model using other k-1 subsets
    #   Evaluate
    shuffled_indices = default_rng().permutation(len(dataset))
    data_buckets = []
    interval = dataset.shape[0] // k

    for i in range(0, dataset.shape[0] + 1, interval):
        data_buckets.append(dataset[shuffled_indices[i:i + interval]])

    confusion_matrix = np.zeros(shape=(4, 4), dtype=int)

    for i in range(k):
        test = data_buckets.pop(0)  # Remove from front
        model, _ = decision_tree_learning(np.concatenate(data_buckets), 0)  # Train on remaining
        data_buckets.append(test)  # Add back to end
        confusion_matrix = update_confusion_matrix(confusion_matrix, model, test)

    accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    recall, precision, f1 = calc_recall_precision(confusion_matrix)
    # print(confusion_matrix)
    # print(np.sum(confusion_matrix))
    print("Accuracy: " + str(100 * accuracy))
    print("Recall: " + str(recall))
    print("Precision: " + str(precision))
    print("F1 score: " + str(f1))


def draw_node(model, ax, props, x, y, depth_curr, depth):
    if model.leaf:
        ax.text(x - 1, y + 0.1, f'Leaf: {str(int(model.value))}', fontsize=6,
                verticalalignment='top', bbox=props)
    else:
        ax.text(x - 4, y + 0.1, f'X{model.attribute} < {model.value}', fontsize=6,
                verticalalignment='top', bbox=props)

        draw_node(model.left, ax, props, x - (8 * depth) / pow(2, depth_curr), y - 1, depth_curr + 1, depth)
        draw_node(model.right, ax, props, x + (8 * depth) / pow(2, depth_curr), y - 1, depth_curr + 1, depth)
        plt.plot([x, x + (8 * depth) / pow(2, depth_curr)], [y, (y - 1)])
        plt.plot([x, x - (8 * depth) / pow(2, depth_curr)], [y, (y - 1)])


def draw_tree(model, depth):
    # TODO: Draw a tree w/ matplotlib
    _, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    draw_node(model, ax, props, 0.5, 0.5, 1, depth)
    plt.yticks([0, 10])
    plt.show()


# TODO: create a tree from the dataset (only clean dataset needed), then draw it
def visualise_tree(data):
    tree, depth = decision_tree_learning(data, 0)
    draw_tree(tree, depth)


if __name__ == "__main__":
    print("Clean")
    data_clean = np.loadtxt("./wifi_db/noisy_dataset.txt")
    # create_pruned_tree(data_clean)

    visualise_tree(data_clean)
    # k_fold_cross_validation(data_clean, 10)
    # print()
    # print("Noisy")
    # data_noisy = np.loadtxt("./wifi_db/noisy_dataset.txt")
    # k_fold_cross_validation(data_noisy, 10)

    # train_clean, test_clean = split_dataset(data_clean, 0.2)
    # root_clean, depth_clean = decision_tree_learning(train_clean, 0)
    # print("Clean")
    # draw_tree(root_clean, depth_clean)

    # evaluate(root_clean, test_clean)
    #
    # data_noisy = np.loadtxt("./wifi_db/noisy_dataset.txt")
    # train_noisy, test_noisy = split_dataset(data_noisy, 0.2)
    # root_noisy, depth_noisy = decision_tree_learning(train_noisy, 0)
    # print("Noisy")
    # evaluate(root_noisy, test_noisy)
