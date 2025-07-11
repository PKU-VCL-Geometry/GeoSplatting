from time import time

from rfstudio.graphics import Points

if __name__ == '__main__':
    points = Points.randn((100000, 3))
    start = time()
    distances = points.k_nearest(k=3)[0]
    print(f'take {time()-start:.3f} seconds')

    points = Points.zeros(3).translate(-0.5)
    assert (points.positions == -0.5).all()
    assert (points.scale(2).positions == -1).all()
