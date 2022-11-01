decision_tree.py takes in several arguments as input:

The relative .txt file path for the dataset (required, no argument tag)
-k: k value for k-fold cross validation, defaults to 10
-prune: prunes every decision tree after learning
-draw: draws unpruned decision tree learnt on whole dataset

Example:

    python3 decision_tree.py './wifi_db/clean_dataset.txt' -k 8 -prune -draw

To run the code on lab machines, execute the following commands in order:

    source /vol/lab/intro2ml/venv/bin/activate
    python3 decision_tree.py {args}