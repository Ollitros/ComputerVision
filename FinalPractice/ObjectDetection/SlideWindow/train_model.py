from FinalPractice.ObjectDetection.SlideWindow.model import InceptionNN
from FinalPractice.ObjectDetection.SlideWindow import utils
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


# Load dataset
train_x, train_y, test_x, test_y = utils.load_dataset()

# Label preprocessing
train_y, test_y = utils.label_encoding(train_y, test_y)

input_shape = (256, 256, 1)

# Training model
model = InceptionNN(input_shape=input_shape, alpha=1.0)
print(model.summary())

# checkpoint
filepath = "models/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.99, min_lr=0.00001)

model.fit(train_x, train_y, batch_size=3, epochs=100, validation_data=(test_x, test_y),
          callbacks=[learning_rate_reduction, checkpoint])
model.save_weights('models/model_weights.h5')

model.load_weights('models/model_weights.h5')
# model.load_weights('models/weights-0.73.h5')
accuracy = model.evaluate(test_x, test_y)
print(accuracy)

prediction = model.predict(test_x)

for x, y in zip(test_x, prediction):
    print(y)
    utils.show_images(x)
