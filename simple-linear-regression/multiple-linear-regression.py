import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("../data/50_Startups.csv")
# select column X
X = dataset.iloc[:, :-1].values
# select column y
y = dataset.iloc[:, -1].values