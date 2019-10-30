import numpy as np

import matplotlib.pyplot as plt
from ml_tps.tp3.space_generation import BooleanPlane, Point


def e1(epoch=5000, points_count=100, eta=0.05, min_error=0.0):

    unit = 7

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    plane = BooleanPlane()

    xs = np.arange(unit + 1) * 5 / unit
    ys = np.arange(unit + 1) * 5 / unit

    points = list(filter(lambda p: -1.5 < p.z < 6, [plane.update_z(Point(x, y, 0)) for y in ys for x in xs]))

    xss = [p.x for p in points]
    yss = [p.y for p in points]
    zss = [p.z for p in points]

    points = [Point.generate_point() for _ in range(points_count)]
    pointsUP = list(filter(plane.classify, points))
    pointsDO = list(filter(lambda p: not plane.classify(p), points))

    xsUP = [point.x for point in pointsUP]
    ysUP = [point.y for point in pointsUP]
    zsUP = [point.z for point in pointsUP]
    xsDN = [point.x for point in pointsDO]
    ysDN = [point.y for point in pointsDO]
    zsDN = [point.z for point in pointsDO]

    ax.plot_trisurf(xss, yss, zss, color="#FFFFFFA0")
    ax.scatter(xsUP, ysUP, zsUP, c="#FF0000")
    ax.scatter(xsDN, ysDN, zsDN, c="#FF00FF")
    plt.show()

    # override = {"epoch" : EPOCH,
    #            "input":
    #                [{"pattern": [point.x, point.y], "response": 1 if plane.classify(point) else -1} for point in points]
    #            }
    # net = mainNet(override)

    from ml_tps.algorithms.simple_perceptron import SimplePerceptron, Pattern
    patterns = [Pattern([point.x, point.y, point.z], 1 if plane.classify(point) else -1) for point in points]
    perceptron = SimplePerceptron(3, min_error, eta, patterns, epoch)
    perceptron.train()

    pointsUPn = \
        list(
            filter(
                lambda point: perceptron.get_value(
                    Pattern([point.x, point.y, point.z], 1 if plane.classify(point) else -1)) == 1, points))
    pointsDOn = \
        list(
            filter(
                lambda point: perceptron.get_value(
                    Pattern([point.x, point.y, point.z], 1 if plane.classify(point) else -1)) == -1, points))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_trisurf(xss, yss, zss, color="#000000FF")

    xsUPn = [point.x for point in pointsUPn]
    ysUPn = [point.y for point in pointsUPn]
    zsUPn = [point.z for point in pointsUPn]
    xsDNn = [point.x for point in pointsDOn]
    ysDNn = [point.y for point in pointsDOn]
    zsDNn = [point.z for point in pointsDOn]

    ax.scatter(xsUPn, ysUPn, zsUPn, c="#00FF00FF")
    ax.scatter(xsDNn, ysDNn, zsDNn, c="#00FFFFFF")
    # print(len(net.errors))
    plt.show()
    print(f"Final error {perceptron.error}")
    print(f"Epochs run {perceptron.last_train_i}")