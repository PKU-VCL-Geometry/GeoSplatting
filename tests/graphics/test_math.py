import torch

from rfstudio.graphics import Points
from rfstudio.graphics.math import (
    PI,
    get_projection,
    get_radian_distance,
    get_random_normal_from_hemisphere,
    get_random_normal_from_sphere,
    get_uniform_normal_from_hemisphere,
    get_uniform_normal_from_sphere,
)
from rfstudio.visualization import Visualizer


def quantitative() -> None:
    points = torch.zeros((100, 100, 3))
    points[:, :, 0] = torch.arange(100).view(1, -1)
    points[:, :, 1] = torch.arange(100).view(-1, 1)
    proj = get_projection(points, plane='pca')[1]
    assert torch.allclose(proj, get_projection(points, plane='xy')[1])

    a = torch.linspace(-4 * PI, 4 * PI, 1000).view(-1, 1)
    offset = torch.rand(100) * 2 * PI - PI
    assert torch.allclose(get_radian_distance(a + offset, a), offset.abs(), atol=1e-5, rtol=1e-5)

def qualitative() -> None:
    rand_sphere_normal = Points(positions=get_random_normal_from_sphere(1024))
    rand_hemisphere_normal = Points(positions=get_random_normal_from_hemisphere(1024, direction=(1, 2, 3)))
    uni_sphere_normal = Points(positions=get_uniform_normal_from_sphere(1024))
    uni_hemisphere_normal = Points(positions=get_uniform_normal_from_hemisphere(1024, direction=(-4, -5, -9)))
    with Visualizer().customize() as handle:
        handle['rand_sphere_normal'].show(rand_sphere_normal).configurate(point_size=0.02)
        handle['rand_hemisphere_normal'].show(rand_hemisphere_normal).configurate(point_size=0.02)
        handle['uni_sphere_normal'].show(uni_sphere_normal).configurate(point_size=0.02)
        handle['uni_hemisphere_normal'].show(uni_hemisphere_normal).configurate(point_size=0.02)

if __name__ == '__main__':
    quantitative()
    qualitative()
