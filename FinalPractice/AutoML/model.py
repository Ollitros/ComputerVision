import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model
from FinalPractice.AutoML.ulits.QLearningAgent import QLearningAgent


class Environment:
    def __init__(self, max_filters):

        self.actions = None
        self.states = None
        self.max_filters = max_filters

    @property
    def n(self):
        return self.actions.size

    def reset(self):
        self.actions = np.zeros(4)
        self.states = np.arange(1, self.max_filters)

        return self.states


class Controller:
    def __init__(self, max_filters):
        self.env = Environment(self.max_filters)
        self.env.reset()
        n_actions = self.env.n
        print(n_actions)
        print(self.env)
        self.max_filters = max_filters
        self.agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99, get_legal_actions=lambda s: range(n_actions))

    def train(self):

        total_reward = 0.0
        s = self.env.reset()

        a = self.agent.get_action(s)  # <get agent to pick action given state s>

        next_s, r, done = self.env.step(a)

        # <train (update) agent for state s>
        self.agent.update(s, a, r, next_s)

        total_reward += r

        return total_reward

    def step(self):

        s = self.env.reset()
        prediction = self.agent.get_action(s)

        return prediction


class AutoML:

    def __init__(self, max_filters=64):
        self.max_filters = max_filters
        self.controller = Controller(self.max_filters)

    def generator(self, inputs):

        filters = self.controller.step()
        x = Conv2D(filters, (3, 3), activation='relu')(inputs)

        return x

    def model_block(self, input_shape, num_classes):

        inputs = Input(shape=input_shape)

        generated = self.generator(inputs)

        x = Flatten()(generated)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, x_test, y_test, batch_size, train_epochs, search_epochs, input_shape, num_classes):

        for i in search_epochs:
            model = self.model_block(input_shape=input_shape, num_classes=num_classes)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=train_epochs, verbose=1)
            accuracy = model.evaluate(x_test, y_test)
            print("Search Step - ", i+1, "||| accuracy - ", accuracy)

