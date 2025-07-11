from __future__ import annotations

from rfstudio.data.downloader import BlenderDownloader, Mip360Downloader
from rfstudio.engine.task import TaskGroup

if __name__ == '__main__':
    TaskGroup(
        blender=BlenderDownloader(),
        mip360=Mip360Downloader(),
    ).run()
