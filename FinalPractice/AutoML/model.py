import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model
from FinalPractice.AutoML.ulits.QLearningAgent import QLearningAgent


class Environment:
    """
        Create environment for agent as a field of int`s where each correspond to picked layer
    """
    def __init__(self):

        self.actions = None
        self.states = None

    @property
    def n(self):
        return self.actions.size

    def reset(self):
        self.actions = np.arange(0, 3)
        self.states = np.zeros(3)

        return self.states

    def step(self, action, step=0, accuracy=0, best_accuracy=0):
        reward = 0

        if accuracy is not None:
            if best_accuracy > accuracy:
                reward = -5
                self.states[-1] = action
            elif best_accuracy < accuracy:
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
    """
        An agent which tries to learn which model configuration is the best for particular dataset.
        First steps performs in train function, then the last one computed in update function, where
        environment decide which accuracy is satisfied and which is not.
    """
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

    def update(self, accuracy, best_accuracy):
        next_s, r = self.env.step(self.update_action, accuracy=accuracy, best_accuracy=best_accuracy)

        self.agent.update(self.update_state, self.update_action, r, next_s)

        total_reward = self.total_reward + r
        self.total_reward = 0

        return total_reward

    # Predict function exists as evaluation for agent
    def predict(self):

        s = self.env.reset()
        prediction = []

        for i in range(self.n_layers):
            a = self.agent.get_action(s)  # <get agent to pick action given state s>
            prediction.append(a)

        return prediction


class AutoML:

    def get_layer(self, i):

        if i == 0:
            return Conv2D(4, (3, 3), activation='relu', padding='same')
        elif i == 1:
            return Conv2D(8, (3, 3), activation='relu', padding='same')
        else:
            return Conv2D(16, (3, 3), activation='relu', padding='same')

    def generator(self, inputs, controller):

        layers = controller.train()
        x = self.get_layer(layers[0])(inputs)
        x = self.get_layer(layers[1])(x)
        x = self.get_layer(layers[2])(x)
        return x, layers

    def model_block(self, input_shape, num_classes, controller):

        inputs = Input(shape=input_shape)

        generated, layers = self.generator(inputs, controller)

        x = Flatten()(generated)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model, layers

    def fit(self, x_train, y_train, x_test, y_test, batch_size, train_epochs, search_epochs, input_shape, num_classes,
            best_accuracy=0, option=False):

        """
            best_accuracy = 0.65  # If you want model does actions to overperform particular accuracy - tune threshold
            option = False        # If you want model does actions to overperform best accuracy choose - option to True
        """

        controller = Controller(n_layers=3)
        for i in range(search_epochs):
            model, layers = self.model_block(input_shape=input_shape, num_classes=num_classes, controller=controller)
            # print(model.summary())
            model.fit(x_train, y_train, batch_size=batch_size, epochs=train_epochs, verbose=0)
            accuracy = model.evaluate(x_test, y_test, verbose=0)
            accuracy = accuracy[1]
            total_reward = controller.update(accuracy, best_accuracy)
            print(layers)
            print("Search Step - ", i + 1, "||| accuracy - ", accuracy, "||| total reward  - ", total_reward)

            if option:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy

        prediction = controller.predict()
        print("Final prediction after training - ", prediction)

