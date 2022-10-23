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

    filter_attr = data[:, [best_attribute, -1]]  # take col for attribute being used for split and room col
    filter_val_gr = filter_attr[
        filter_attr[:, 0] >= best_value]  # filter for values greater than or equal to split val (right)
    filter_val_le = filter_attr[filter_attr[:, 0] < best_value]  # same but for less than (left)
    roomsgr, grcount = np.unique(filter_val_gr[-1], return_counts=True)
    roomsle, lecount = np.unique(filter_val_le[-1], return_counts=True)

    most_likely_room_gr = roomsgr[np.argmax(grcount)]
    most_likely_room_le = roomsle[np.argmax(lecount)]

    # not sure what way round this is:
    most_likely_r = most_likely_room_gr
    most_likely_l = most_likely_room_le

    return best_attribute, best_value, best_l_split, best_r_split, most_likely_r, most_likely_l
    
    
    
def decision_tree_learning(training_dataset, depth):
    y = training_dataset[:, -1]

    if np.min(y) == np.max(y):  # All samples have same categorical value
        return Node(value=y[0], leaf=True), depth

    attr, val, l_dataset, r_dataset, mlr, mll= find_split(training_dataset)
    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
    node = Node(attribute=attr, value=val, left=l_branch, right=r_branch)
    node.left.most_likely_room = mll
    node.right.most_likely_room = mlr

    return node, max(l_depth, r_depth)