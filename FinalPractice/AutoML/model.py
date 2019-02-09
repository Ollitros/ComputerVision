import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model
from FinalPractice.AutoML.ulits.QLearningAgent import QLearningAgent


class Environment:
    def __init__(self):

        self.actions = None
        self.states = None

        self.accuracy = 0

    @property
    def n(self):
        return self.actions.size

    def reset(self):
        self.actions = np.arange(0, 3)
        self.states = np.zeros(3)

        return self.states

    def step(self, action, step, accuracy=None):
        reward = 0

        if accuracy is not None:
            if self.accuracy > accuracy:
                reward = -10
                self.states[-1] = action
            elif self.accuracy < accuracy:
                reward = 10
                self.states[-1] = action
            else:
                reward = -1
                self.states[-1] = action
        else:
            reward = 1
            self.states[step] = action

        return self.states, reward


class Controller:
    def __init__(self, n_layers):
        self.env = Environment()
        self.env.reset()
        self.n_layers = n_layers
        n_actions = self.env.n
        self.agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99, get_legal_actions=lambda s: range(n_actions))
        self.update_state = None
        self.update_action = None
        self.total_reward = 0

    def train(self):

        s = self.env.reset()
        actions = []

        for i in range(self.n_layers):
            a = self.agent.get_action(s)  # <get agent to pick action given state s>
            actions.append(a)

            if i == self.n_layers-1:
                self.update_action = a
                break
            else:
                next_s, r = self.env.step(a, step=i)

                # <train (update) agent for state s>
                self.agent.update(s, a, r, next_s)
                s = next_s

                self.total_reward += r

        self.update_state = s

        return actions

    def update(self, accuracy):
        next_s, r = self.env.step(self.update_action, accuracy)

        self.agent.update(self.update_state, self.update_action, r, next_s)

        total_reward = self.total_reward
        self.total_reward = None

        return total_reward

    # Step function should be changed, don`t even touch
    def step(self):

        s = self.env.reset()
        prediction = self.agent.get_action(s)

        return prediction


class AutoML:

    def __init__(self, max_filters=64):
        self.max_filters = max_filters

    def get_layer(self, i):

        if i == 0:
            return Conv2D(16, (3, 3), activation='relu', padding='same')
        elif i == 1:
            return Conv2D(32, (3, 3), activation='relu', padding='same')
        else:
            return Conv2D(64, (3, 3), activation='relu', padding='same')

    def generator(self, inputs, controller):

        layers = controller.train()
        print(layers)
        x = self.get_layer(layers[0])(inputs)
        x = self.get_layer(layers[1])(x)
        x = self.get_layer(layers[2])(x)
        return x

    def model_block(self, input_shape, num_classes, controller):

        inputs = Input(shape=input_shape)

        generated = self.generator(inputs, controller)

        x = Flatten()(generated)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, x_test, y_test, batch_size, train_epochs, search_epochs, input_shape, num_classes):
        controller = Controller(n_layers=3)
        for i in range(search_epochs):
            model = self.model_block(input_shape=input_shape, num_classes=num_classes, controller=controller)
            print(model.summary())
            model.fit(x_train, y_train, batch_size=batch_size, epochs=train_epochs, verbose=1)
            accuracy = model.evaluate(x_test, y_test)
            print("Search Step - ", i+1, "||| accuracy - ", accuracy)
            total_reward = controller.update(accuracy)
            print("Search Step - ", i + 1, "||| total reward  - ", total_reward)

