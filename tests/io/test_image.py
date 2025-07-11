import pathlib

from rfstudio.io import load_float32_image

if __name__ == '__main__':
    path = pathlib.Path('data') / 'blender' / 'chair' / 'test'
    assert path.exists()
    assert load_float32_image(path / 'r_0_depth_0000.png').shape == (800, 800, 4)
    assert load_float32_image(path / 'r_0.png', alpha_color=(1, 1, 1)).shape == (800, 800, 3)
    assert load_float32_image(path / 'r_0.png', scale_factor=0.5).shape == (400, 400, 4)
