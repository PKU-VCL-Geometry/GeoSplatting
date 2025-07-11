from __future__ import annotations

import contextlib
import shutil
from http.server import HTTPServer, SimpleHTTPRequestHandler
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import rich
from rich import box, style
from rich.panel import Panel
from rich.table import Table

_ASSETS_DIR: Path = files('rfstudio') / 'assets' / 'web'


@contextlib.contextmanager
def open_webserver(assets_name: str, *, host: str = '0.0.0.0', port: int = 6789) -> Iterator[Path]:
    tmpdir = None
    server = None
    try:
        tmpdir = TemporaryDirectory()
        tmppath = Path(tmpdir.__enter__()) / 'web'
        assert tmppath == Path(shutil.copytree(_ASSETS_DIR / assets_name, tmppath))
        shutil.copy(_ASSETS_DIR / 'vite.svg', tmppath)

        yield tmppath

        class _Handler(SimpleHTTPRequestHandler):

            def __init__(self, request, client_address, server):
                super(_Handler, self).__init__(request, client_address, server, directory=str(tmppath))

            def log_message(self, format, *args):
                pass

        server = HTTPServer((host, port), _Handler)

        http_url = f"[bold][yellow]http://{host}:{port}[/yellow][/bold]"
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("[bold]HTTP[/bold]", http_url)
        rich.print(Panel(table, title="[bold]vis 3dgs[/bold]", expand=False))

        server.serve_forever()

    except KeyboardInterrupt:
        pass
    finally:
        if tmpdir is not None:
            tmpdir.__exit__(None, None, None)
        if server is not None:
            server.server_close()
