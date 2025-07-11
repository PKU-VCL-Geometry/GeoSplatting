import time

import numpy as np

from rfstudio.ui import console


def test_status():
    with console.status('Anything...'):
        time.sleep(4)
    time.sleep(1)
    with console.status('Anything...', screen=True):
        time.sleep(4)

def test_progress():
    with console.progress('Progressing...') as ptrack:
        for i in ptrack(range(10)):
            time.sleep(1)

def test_screen():
    with console.screen(desc='Prepare...', refresh_interval=0.5) as handle:
        handle.set_layout(
            handle.cols[3, 1](
                handle.table['left'],
                handle.table['right']
            )
        )
        handle.progress['i'].update(curr=0, total=3)
        for i in range(3):
            handle.progress[f'k{i}'].update(curr=0, total=10)
            handle.table['left'].update(i=i)
            for k in range(10):
                time.sleep(1)
                handle.progress[f'k{i}'].update(curr=k+1, total=10)
                handle.table['right'].update(i=i, k=k*0.1)
                handle.sync()
            handle.progress['i'].update(curr=i+1, total=3)
            handle.sync()

def test_plot():
    with console.screen(refresh_interval=1) as handle:
        handle.set_layout(
            handle.cols[2, 3](
                handle.table['status'],
                handle.plot['example']
            )
        )
        timestep = 0
        while True:
            y = {'sin(x)': np.sin(timestep)}
            if timestep > 5:
                y['cos(x)'] = np.cos(timestep)
            handle.plot['example'].update(
                x=timestep,
                xlabel='step',
                y=y,
                ylabel='value',
                xlim=(None, 10),
                ylim=(-1, 1)
            )
            handle.progress['computing'].update(curr=timestep / 10)
            handle.table['status'].update(x=timestep, **y)
            handle.sync()
            time.sleep(0.5)
            timestep += 0.3
            if timestep > 10:
                break
        handle.hold('Finished.')


if __name__ == '__main__':
    test_status()
    test_progress()
    test_screen()
    test_plot()
