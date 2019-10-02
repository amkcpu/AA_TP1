import numpy as np


def step(w, pattern_input):
    return np.dot(w, pattern_input)


class Pattern:

    def __init__(self, raw_pattern, response):
        self.input = np.array([0] + raw_pattern) / 2.5 - 1
        self.response = response

    def __repr__(self):
        return f"[{self.input}, {self.response}]"

    def __str__(self):
        return self.__repr__()



class SimplePerceptron:

    def __init__(self, size, error, eta, patterns, epochs):
        self.w = np.random.random(size+1)
        self.error_min_accepted = error
        self.eta = eta
        self.patterns = patterns
        self.epochs = epochs
        self.error = sum(
            [abs(pattern.response - np.sign(step(self.w, pattern.input))) for pattern in self.patterns]) / len(
            self.patterns)
        self.min_error = self.error
        self.min_w = self.w
        self.last_train_i = -1

    def train(self):
        i = 0
        while self.error > self.error_min_accepted and self.epochs > i:
            np.random.shuffle(self.patterns)

            #if (i - 1) % (self.epochs / 5) == 0:
            #    self.w = np.random.random(len(self.w))

            for pattern in self.patterns:
                output = step(self.w, pattern.input)
                delta_w = self.eta * (pattern.response - np.sign(output)) * pattern.input
                self.w += delta_w

            self.error = sum([abs(pattern.response - np.sign(step(self.w, pattern.input))) for pattern in self.patterns]) / len(self.patterns)
            if self.error < self.min_error:
                self.min_error = self.error
                self.min_w = self.w.copy()
            i += 1

        self.error = self.min_error
        self.w = self.min_w
        self.last_train_i = i

    def get_value(self, pattern: Pattern):
        return np.sign(step(self.w, pattern.input))

