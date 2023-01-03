import pandas as pd
from mlflow_logging import log_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv("./data/train_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# log model training and evaluation with mlflow
log_model(model, X_train, X_test, y_train, y_test, "Random Forest Regressor", "regression")


