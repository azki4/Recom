import pandas as pd
from surprise import Dataset, Reader

# Load the MovieLens dataset
data = Dataset.load_builtin('ml-100k')

from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the data into a trainset and testset
trainset, testset = train_test_split(data, test_size=0.25)

# Use KNNBasic algorithm
algo = KNNBasic()

# Train the algorithm on the trainset
algo.fit(trainset)

# Test the algorithm on the testset
predictions = algo.test(testset)

# Calculate and print the RMSE
accuracy.rmse(predictions)

from surprise.model_selection import cross_validate

# Perform cross-validation
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
