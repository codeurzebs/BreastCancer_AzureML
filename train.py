# Copyright (c) 2024, codeurzebs
# All rights reserved.
#
# This file is part of the project hosted at https://github.com/codeurzebs
#
# The source code is subject to the terms and conditions defined in the
# file 'LICENSE.txt', which is part of this source code package.

# Importing necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
import argparse
import os
import numpy as np
# from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from sklearn import datasets

# Get the Azure ML run context
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument('--n_estimators', type=int, default=10, help="The number of boosting stages to perform.")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate shrinks the contribution of each tree.")
    parser.add_argument('--max_depth', type=int, default=1, help="The maximum depth of the individual regression estimators.")

    args = parser.parse_args()

    # Log hyperparameters to Azure ML
    run.log("Number of estimators:", int(args.n_estimators))
    run.log("Learning Rate:", float(args.learning_rate))
    run.log("Maximum Depth of Tree:", int(args.max_depth))

    # Load breast cancer dataset
    data = datasets.load_breast_cancer()
    x, y = data.data, data.target

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=12345)

    # Train the Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=12345
    ).fit(x_train, y_train)

    # Calculate and log accuracy
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", float(accuracy))

    # Save the trained model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
