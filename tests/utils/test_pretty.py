import pathlib

from rich.console import Console

from rfstudio.utils.pretty import P

if __name__ == '__main__':
    c = Console()

    c.print(P@'anything = {2}')

    d1 = {
        'a': 1,
        'c': True
    }
    d2 = {
        'a': 1,
        'b': 1.2,
        'c': True,
        'd': pathlib.Path('/') / 'usr' / 'lib'
    }

    c.print(P@'anything = {d1} something = {d2}')

    c.log(P(d2))
