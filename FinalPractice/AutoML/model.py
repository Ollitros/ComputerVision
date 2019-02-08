import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model


class Environment:
    def __init__(self, game_mode=1):

        self.actions = None
        self.states = None

    @property
    def n(self):
        return self.actions.size

    def reset(self):
        self.actions = np.zeros(4)
        self.states = np.zeros(4)

        return self.states


class Controller:

    def train(self):

        env = Environment()
        env.reset()
        n_actions = env.n
        print(n_actions)
        print(env)

        agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99,
                               get_legal_actions=lambda s: range(n_actions))

        def play_and_train(env, agent, t_max=10 ** 4):
            """This function should
            - run a full game, actions given by agent.getAction(s)
            - train agent using agent.update(...) whenever possible
            - return total reward"""
            total_reward = 0.0
            s = env.reset()

            for t in range(t_max):
                a = agent.get_action(s)  # <get agent to pick action given state s>

                action = env.states
                if action[a] == 1:
                    continue

                next_s, r, done = env.step(a)

                # <train (update) agent for state s>
                agent.update(s, a, r, next_s)

                s = next_s
                total_reward += r
                if done:
                    break

            return total_reward

        rewards = []
        for i in range(6000):
            rewards.append(play_and_train(env, agent))
            if i % 1000 == 0:
                clear_output(True)
                print("mean reward", np.mean(rewards[-100:]))
                plt.plot(rewards)
                plt.show()

    def predict(self):

        prediction =

        return prediction



class AutoML:
    def __init__(self, max_filters=64):
        self.max_filters = max_filters
        self.controller = Controller()

    def generator(self, inputs):

        filters = self.controller.predict()
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
            model.fit(x_train, y_train, batch_size=batch_size, epochs=train_epochs, verbose=0)
            accuracy = model.evaluate(x_test, y_test)
            print("Search Step - ", i+1, "||| accuracy - ", accuracy)
