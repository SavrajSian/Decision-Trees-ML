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
        return f"Node: Left[{self.left}] - Right[{self.right}]"


def find_split(dataset):
    # Finds the attribute and value with the highest information gain
    # Sort a column -> take two adjacent values mean -> calculate info gain on this mean
    max_gain = 0
    best_attribute, best_value = 0, 0
    best_l_split, best_r_split = [[]], [[]]
    attr = 0
    h_all = entropy(dataset)
    for col in dataset[:, :-1].T:
        sorted_col = np.sort(col)
        for i in range(sorted_col.size - 1):
            mp = (sorted_col[i] + sorted_col[i + 1]) / 2  # Mean of adjacent values
            l_split = dataset[col < mp]  # dataset to left of midpoint
            r_split = dataset[col >= mp]  # dataset to right of midpoint
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


def tree_depth(node):
    if node.leaf:
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def evaluate(test_db, trained_tree):
    success = 0
    total = 0
    for sample in test_db:
        x, y = sample[:-1], int(sample[-1])
        result = fit(trained_tree, x)
        total += 1
        if result == y:
            success += 1
    accuracy = 100 * (success / total)
    return accuracy


def update_confusion_matrix(confusion_matrix, model, test):
    for sample in test:
        x, y = sample[:-1], int(sample[-1])
        result = fit(model, x)
        # x-axis predicted labels, y-axis actual labels
        confusion_matrix[(y - 1, result - 1)] += 1
    return confusion_matrix


def calc_recall_precision_f1(confusion_matrix):
    # Calculate recall and precision rates and f1 measures per class (as percentages)
    precision = []
    recall = []
    f1 = []

    for i in range(num_classes):
        tp = confusion_matrix[i][i]
        tp_scaled = 100 * tp  # for percentages
        pr = round(tp_scaled / np.sum(confusion_matrix[:, i], axis=0), 2)
        re = round(tp_scaled / np.sum(confusion_matrix[i]), 2)
        precision.append(pr)
        recall.append(re)
        f1.append(round(2 * pr * re / (pr + re), 2))

    return recall, precision, f1


def decision_tree_learning(training_dataset, depth):
    y = training_dataset[:, -1]

    if np.min(y) == np.max(y):  # All samples have same categorical value
        return Node(value=y[0], leaf=True), depth

    attr, val, l_dataset, r_dataset = find_split(training_dataset)
    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
    node = Node(attribute=attr, value=val, left=l_branch, right=r_branch)

    return node, max(l_depth, r_depth)


def prune(curr_node, train, validation):
    if curr_node.leaf:
        return curr_node

    node = curr_node
    attr, val = node.attribute, node.value

    train_left = train[train[:, attr] < val]
    train_right = train[train[:, attr] >= val]

    node.left = prune(node.left, train_left, validation)
    node.right = prune(node.right, train_right, validation)

    if not node.left.leaf or not node.right.leaf:
        return node

    rooms, counts = np.unique(train[:, -1], return_counts=True)
    majority = rooms[np.argmax(counts)]

    new_node = Node(value=majority, leaf=True)

    acc_before = evaluate(validation, node)
    acc_after = evaluate(validation, new_node)

    if acc_before > acc_after:
        return node

    return new_node


def pruned_tree_learning(train, validation):
    model, _ = decision_tree_learning(train, 0)
    pruned_model = prune(model, train, validation)
    while pruned_model != model:
        model = pruned_model
        pruned_model = prune(model, train, validation)
    return pruned_model


def print_metrics(confusion_matrix):
    accuracy = 100 * (np.diag(confusion_matrix).sum() / confusion_matrix.sum())
    recall, precision, f1 = calc_recall_precision_f1(confusion_matrix)

    print("Confusion Matrix:")
    print(confusion_matrix.round(2))
    print("Accuracy: " + str(round(accuracy, 2)))
    print("Recall: " + str(recall))
    print("Precision: " + str(precision))
    print("F1 score: " + str(f1))


def k_fold_cross_validation(dataset, k):
    # Split data k equal ways
    # Repeat k times:
    #   Take a subset of data to test
    #   Train model using other k-1 subsets
    #   Evaluate
    shuffled_indices = default_rng().permutation(len(dataset))
    data_buckets = []
    sum_depths = 0
    samples = dataset.shape[0]

    # i.e. 10-fold with 200 samples will make buckets of 20
    interval = samples // k

    confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype=int)

    # Make of list of k buckets
    for i in range(0, samples + 1, interval):
        data_buckets.append(dataset[shuffled_indices[i:i + interval]])

    for i in range(k):
        test = data_buckets.pop(0)  # Remove from front
        train = np.concatenate(data_buckets)
        model, depth = decision_tree_learning(train, 0)

        sum_depths += depth
        data_buckets.append(test)  # Add back to end
        confusion_matrix = update_confusion_matrix(confusion_matrix, model, test)

    avg_confusion_matrix = confusion_matrix / k
    avg_depth = sum_depths / k
    return avg_confusion_matrix, avg_depth


def nested_k_fold_cross_validation(dataset, k):
    # Split data k equal ways
    # Repeat k times:
    #   Take a subset of data to test
    #   Repeat k - 1 times:
    #       Take a subset of data to validate
    #       Train model using the remaining k-2 subsets + the validation set (for evaluation)
    #       Evaluate model on the test set
    shuffled_indices = default_rng().permutation(len(dataset))
    data_buckets = []
    sum_depths = 0
    samples = dataset.shape[0]

    interval = samples // k

    confusion_matrix = np.zeros(shape=(num_classes, num_classes), dtype=float)

    for i in range(0, samples + 1, interval):
        data_buckets.append(dataset[shuffled_indices[i:i + interval]])

    for i in range(k):
        test = data_buckets.pop(0)
        for j in range(k - 1):
            validation = data_buckets.pop(0)
            train = np.concatenate(data_buckets)
            model = pruned_tree_learning(train, validation)
            depth = tree_depth(model)

            sum_depths += depth
            data_buckets.append(validation)
            confusion_matrix = update_confusion_matrix(confusion_matrix, model, test)

        data_buckets.append(test)

    iterations = k * (k - 1)
    avg_confusion_matrix = confusion_matrix / iterations
    avg_depth = sum_depths / iterations
    return avg_confusion_matrix, avg_depth


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
    _, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    draw_node(model, ax, props, 0.5, 0.5, 1, depth)
    plt.yticks([0, 10])
    plt.show()


def visualise_tree(dataset):
    tree, depth = decision_tree_learning(dataset, 0)
    draw_tree(tree, depth)


if __name__ == "__main__":
    data_clean = np.loadtxt("./wifi_db/clean_dataset.txt")
    data_noisy = np.loadtxt("./wifi_db/noisy_dataset.txt")

    labels = np.unique(data_clean[:, -1])
    num_classes = len(labels)

    print("Drawing tree")
    visualise_tree(data_clean)

    print("Clean\n")
    conf_matrix, average_depth = k_fold_cross_validation(data_clean, 10)
    print_metrics(conf_matrix)
    print(f"Average tree depth: {average_depth}")

    print("\nNoisy\n")
    conf_matrix, average_depth = k_fold_cross_validation(data_noisy, 10)
    print_metrics(conf_matrix)
    print(f"Average tree depth: {average_depth}")

    print("\nClean w/ pruning\n")
    conf_matrix, average_depth = nested_k_fold_cross_validation(data_clean, 10)
    print_metrics(conf_matrix)
    print(f"Average tree depth: {average_depth}")

    print("\nNoisy w/ pruning\n")
    conf_matrix, average_depth = nested_k_fold_cross_validation(data_noisy, 10)
    print_metrics(conf_matrix)
    print(f"Average tree depth: {average_depth}")
