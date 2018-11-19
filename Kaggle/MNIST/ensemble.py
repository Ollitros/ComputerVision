import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from collections import Counter


# Load model from json and weights
def load(weights):
    path = 'data/models/model.json'
    json_file = open(path, 'r')
    # json_file = open(self.path_to_model_json, 'r')
    print('in preidcuit')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weights)
    print("Loaded model from disk")

    return model


# Load data from local folder
train_dataset = pd.read_csv('data/train.csv')
test_dataset = pd.read_csv('data/test.csv')

# Split data on labels and features
train_y = train_dataset['label']
train_x = train_dataset.drop(['label'], axis=1)

# Get from DataFrames values and reshape them
train_y = train_y.values
train_x = train_x.values.reshape(-1, 28, 28, 1)
test_x = test_dataset.values.reshape(-1, 28, 28, 1)

# Transform targets into one-hot vector
train_y = to_categorical(train_y, num_classes=10)

# Create test data and make data normalization
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1)


weights1 = 'data/models/model_1.h5'
weights2 = 'data/models/model_2.h5'
weights3 = 'data/models/model_3.h5'
weights4 = 'data/models/model_4.h5'
weights5 = 'data/models/model_5.h5'
weights6 = 'data/models/model_6.h5'
weights7 = 'data/models/model_7.h5'
weights8 = 'data/models/model_8.h5'
weights9 = 'data/models/model_9.h5'
weights10 = 'data/models/model_10.h5'

model1 = load(weights1)
model2 = load(weights2)
model3 = load(weights3)
model4 = load(weights4)
model5 = load(weights5)
model6 = load(weights6)
model7 = load(weights7)
model8 = load(weights8)
model9 = load(weights9)
model10 = load(weights10)


# Prediction by averaging predictions

# predictions1 = model1.predict(test_x)
# predictions2 = model2.predict(test_x)
# predictions3 = model3.predict(test_x)
# predictions4 = model4.predict(test_x)
# predictions5 = model5.predict(test_x)
# predictions6 = model6.predict(test_x)
# predictions7 = model7.predict(test_x)
# predictions8 = model8.predict(test_x)
# predictions9 = model9.predict(test_x)
# predictions10 = model10.predict(test_x)
#
# prediction = np.average([predictions1, predictions2, predictions3, predictions4, predictions5, predictions6,
#                          predictions7, predictions8, predictions9, predictions10], axis=0)
# pred = np.argmax(prediction, axis=1)


# Prediction by most common prediction in predictions
predictions1 = np.argmax(model1.predict(test_x), axis=1)
predictions2 = np.argmax(model2.predict(test_x), axis=1)
predictions3 = np.argmax(model3.predict(test_x), axis=1)
predictions4 = np.argmax(model4.predict(test_x), axis=1)
predictions5 = np.argmax(model5.predict(test_x), axis=1)
predictions6 = np.argmax(model6.predict(test_x), axis=1)
predictions7 = np.argmax(model7.predict(test_x), axis=1)
predictions8 = np.argmax(model8.predict(test_x), axis=1)
predictions9 = np.argmax(model9.predict(test_x), axis=1)
predictions10 = np.argmax(model10.predict(test_x), axis=1)

pred = []
temp = []
for i in range(len(predictions1)):
    temp.append(predictions1[i])
    temp.append(predictions2[i])
    temp.append(predictions3[i])
    temp.append(predictions4[i])
    temp.append(predictions5[i])
    temp.append(predictions6[i])
    temp.append(predictions7[i])
    temp.append(predictions8[i])
    temp.append(predictions9[i])
    temp.append(predictions10[i])

    common = Counter(temp)
    common = common.most_common(1)
    common = common[0]
    common = common[0]

    pred.append(common)
    temp = []

my_submission = pd.DataFrame({'ImageId': range(1, len(test_x)+1), 'Label': pred})
my_submission.to_csv("data/submission.csv", index=False)