import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)


# Padding function
def pad_sequence(seq, target_length=84):
    if len(seq) < target_length:
        return seq + [0] * (target_length - len(seq))
    elif len(seq) > target_length:
        return seq[:target_length]
    return seq


# Pad all data sequences to length 42
padded_data = [pad_sequence(entry, target_length=42) for entry in data_dict['data']]
data = np.array(padded_data, dtype=np.float32)
labels = np.array(data_dict['labels'])

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
