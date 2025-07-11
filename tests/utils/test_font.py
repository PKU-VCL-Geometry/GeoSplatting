from __future__ import annotations

from pathlib import Path

from rfstudio.io import dump_float32_image
from rfstudio.utils.font import Font

if __name__ == '__main__':
    font = Font.from_name('LinLibertine')
    letters = 'abcdefghijklmnopqrstuvwxyz'
    others = '''!@#$%^&*()-=_+,./;[]{}'":?\\|`~'''
    numbers = '1234567890'
    mask = font.write(letters + letters.upper() + others + numbers, line_height=100)
    dump_float32_image(Path('temp.png'), mask.repeat(1, 1, 3))
