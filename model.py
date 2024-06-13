import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the data
df = pd.read_csv('F_P lozistic.csv')

# Define features (X) and target (Y)
X = df[['Molecular weight','Hydrogen bond acceptor','Hydrogen bond donor','Rotatable bonds','Topological polar surface area','log P','Aromatic bonds']]
Y = df['BBB+/-']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

# Print the shapes of the resulting datasets to verify
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of Y_train: {Y_train.shape}')
print(f'Shape of Y_test: {Y_test.shape}')

# Train a logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Make predictions
pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, pred)
print(f'Accuracy: {accuracy}')

# Save the model and accuracy
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(accuracy, 'accuracy.pkl')
