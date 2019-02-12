from FinalPractice.ObjectDetection.SimpleObjectDetection.model import Inception_resnet
from FinalPractice.ObjectDetection.SimpleObjectDetection import utils


# Load dataset
train_x, train_y, test_x, test_y = utils.load_dataset()

# Label preprocessing
train_y, test_y = utils.label_encoding(train_y, test_y)

input_shape = (256, 256, 1)

# Training model
model = Inception_resnet(input_shape=input_shape, num_classes=3, alpha=1)
model.fit(train_x, train_y, batch_size=3, epochs=10)
model.save_weights('model_weights.h5')
accuracy = model.evaluate(test_x, test_y)
print(accuracy)

prediction = model.predict(test_x)
for x, y in zip(test_x, prediction):
    print(y)
    utils.draw_rect(x, y)