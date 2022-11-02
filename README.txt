decision_tree.py takes in several arguments as input:

The relative .txt file path for the dataset (required, no argument tag)
-k: k value for k-fold cross validation, defaults to 10
-prune: prunes every decision tree after learning
-draw: draws decision tree learnt on whole dataset (unaffected by k, tree is not pruned)

Example:

    python3 decision_tree.py './wifi_db/clean_dataset.txt' -k 8 -prune -draw

Depending on the system you are using, quotation marks surrounding the file path may not work,
in which case do not surround the filepath with quotation marks.

To run the code on lab machines, execute the following commands in order:

    source /vol/lab/intro2ml/venv/bin/activate
    python3 decision_tree.py {args}

Copy and paste the commands in sequence below to test both datasets both without and with pruning (k = 10):

    source /vol/lab/intro2ml/venv/bin/activate
    python3 decision_tree.py './wifi_db/clean_dataset.txt'
    python3 decision_tree.py './wifi_db/noisy_dataset.txt'
    python3 decision_tree.py './wifi_db/clean_dataset.txt' -prune
    python3 decision_tree.py './wifi_db/noisy_dataset.txt' -prune