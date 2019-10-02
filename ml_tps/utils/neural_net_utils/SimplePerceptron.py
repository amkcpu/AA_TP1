import numpy as np


def step(pattern, w):
    return np.dot(pattern.input, w)


class Pattern:

    def __init__(self, raw_pattern, response):
        self.input = np.array([-1] + raw_pattern)
        self.response = response

    def __repr__(self):
        return f"[{self.input}, {self.response}]"

    def __str__(self):
        return self.__repr__()



class SimplePerceptron:

    def __init__(self, size, error, step_size, patterns, iters):
        self.w = np.random.random(size+1)
        self.error_min_accepted = error
        self.step_size = step_size
        self.patterns = patterns
        self.iters = iters
        self.error = sum([abs(pattern.response - np.sign(step(pattern, self.w.T))) for pattern in self.patterns]) / len(
            self.patterns)
        self.min_error = self.error
        self.min_w = self.w

    def train(self):
        i = 0
        while self.error > self.error_min_accepted and self.iters > i:
            np.random.shuffle(self.patterns)

            for pattern in self.patterns:
                output = step(pattern, self.w.T)
                delta_w = self.step_size * (pattern.response - np.sign(output)) * pattern.input.T
                self.w += delta_w

            self.error = sum([abs(pattern.response - np.sign(step(pattern, self.w.T))) for pattern in self.patterns]) / len(self.patterns)
            if self.error < self.min_error:
                self.min_error = self.error
                self.min_w = self.w
            i += 1

        self.error = self.min_error
        self.w = self.min_w

    def get_value(self, pattern: Pattern):
        return np.sign(step(pattern, self.w.T))

