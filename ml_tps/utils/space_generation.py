import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class Point:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def generate_point_by_radius(radius=1):
        phi = np.random.random() * np.pi
        theta = np.random.random() * np.pi
        return Point(x=radius * np.sin(theta) * np.cos(phi),
                     y=radius * np.sin(theta) * np.sin(phi),
                     z=radius * np.cos(theta))

    @staticmethod
    def generate_point(x_range=(0, 5), y_range=(0, 5), z_range=(0, 5)):
        return Point(x=np.random.random() * (x_range[1] - x_range[0]) + x_range[0],
                     y=np.random.random() * (y_range[1] - y_range[0]) + y_range[0],
                     z=np.random.random() * (z_range[1] - z_range[0]) + z_range[0])

    def as_tuple(self):
        return self.x, self.y, self.z


class BooleanPlane:

    def __init__(self, point: Point = Point(2.5, 2.5, 2.5)):
        self.a, self.b, self.c = self.generate_plane(point)

    def classify(self, case: Point) -> bool:
        return self.a * case.x + self.b * case.y + self.c < case.z

    def update_z(self, case: Point) -> Point:
        return Point(case.x, case.y, self.a * case.x + self.b * case.y + self.c)

    @staticmethod
    def generate_plane(fixed_point: Point):
        # c * z = a * x + b * y + d
        # d = c * z - (a * x + b * y)
        # z = (a/c) * x + (b/c) * y + d/c
        # z = e * x + f * y + g
        a, b, c = Point.generate_point_by_radius().as_tuple()
        d = fixed_point.z * c - (a * fixed_point.x + b * fixed_point.y)
        return a / c, b / c, d / c


if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plane = BooleanPlane()
    points = [Point.generate_point() for _ in range(50)]

    xs = [point.x for point in points]
    ys = [point.y for point in points]
    zs = [point.z for point in points]

    ax.scatter(xs, ys, zs)
    plt.show()
