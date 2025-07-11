from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.visualization._director import Animation, Director


@dataclass
class Script(Task):

    director: Director = Director(workspace=Path('temp') / 'demo', resolution='4K')

    export: bool = False

    target_mb: float = 16.0

    def run(self) -> None:
        self.director.to(self.device)

        with self.director.stage('title', duration=60) as sd:
            sd.fade_in(20).fade_out(20)
            sd[0.15:0.25, 0.1:] = Animation.StaticText(
                'GeoSplatting: Towards Geometry',
                bold=True,
                align='left',
            )
            sd[0.27:0.37, 0.1:] = Animation.StaticText(
                'Guided Gaussian Splatting for',
                bold=True,
                align='left',
            )
            sd[0.39:0.49, 0.1:] = Animation.StaticText(
                'Physically-based Inverse Rendering',
                bold=True,
                align='left',
            )

            sd[0.6:0.67, 0.1:] = Animation.StaticText(
                'Anonymous ICCV submission',
                bold=False,
                align='left',
            )
            sd[0.69:0.76, 0.1:] = Animation.StaticText(
                'Paper ID 10617',
                bold=False,
                align='left',
            )

        with self.director.stage('title_comp', duration=60) as sd:
            sd.fade_in(20).fade_out(20)
            sd[0.4:0.6, 0.1:0.9] = Animation.StaticText(
                'Comparison on Decomposition & Relighting',
                bold=True,
            )

        for scene, clip, anim_desc, anim_switch in zip(
            ['hotdog', 'jugs'],
            # ['chair', 'hotdog', 'jugs', 'arm'],
            [
                # (slice(50, -150), slice(100, -100)),
                (slice(50, -50), slice(50, -50)),
                (slice(100, -100), slice(100, -100)),
                # (slice(40, -160), slice(100, -100)),
            ],
            [
                # {
                #     'albedo': (94, slice(0.35, 0.55), slice(0.27, 0.47), slice(0.4, 1.0), slice(0.0, 0.6)),
                #     'relit1': (139, slice(0.33, 0.53), slice(0.63, 0.83), slice(0.0, 0.6), slice(0.0, 0.6)),
                #     'relit2': (232, slice(0.25, 0.45), slice(0.7, 0.9), slice(0.4, 1.0), slice(0.0, 0.6)),
                # },
                {
                    'albedo': (47, slice(0.55, 0.85), slice(0.25, 0.55), slice(0.0, 0.6), slice(0.4, 1.0)),
                    'relit1': (148, slice(0.5, 0.75), slice(0.25, 0.5), slice(0.0, 0.6), slice(0.4, 1.0)),
                    'relit2': (207, slice(0.6, 0.95), slice(0.2, 0.55), slice(0.4, 1.0), slice(0.4, 1.0)),
                },
                {
                    'albedo': (86, slice(0.15, 0.4), slice(0.45, 0.7), slice(0.4, 1.0), slice(0.0, 0.6)),
                    'relit1': (139, slice(0.5, 0.7), slice(0.1, 0.3), slice(0.4, 1.0), slice(0.4, 1.0)),
                    'relit2': (289, slice(0.18, 0.51), slice(0.42, 0.75), slice(0.4, 1.0), slice(0.0, 0.6)),
                },
                # {
                #     'albedo': (72, slice(0.05, 0.3), slice(0.35, 0.6), slice(0.4, 1.0), slice(0.4, 1.0)),
                #     'relit1': (139, slice(0.1, 0.35), slice(0.45, 0.7), slice(0.4, 1.0), slice(0.4, 1.0)),
                #     'relit2': (232, slice(0.25, 0.45), slice(0.7, 0.9), slice(0.4, 1.0), slice(0.0, 0.6)),
                # },
            ],
            [
                # {
                #     'albedo': 40,
                #     'relit1': 220,
                #     'relit2': 390,
                # },
                {
                    'albedo': 30,
                    'relit1': 210,
                    'relit2': 390,
                },
                {
                    'albedo': 40,
                    'relit1': 220,
                    'relit2': 390,
                },
                # {
                #     'albedo': 40,
                #     'relit1': 220,
                #     'relit2': 390,
                # },
            ],
            strict=True,
        ):
            scene_name = ('armadillo' if scene == 'arm' else scene).capitalize()
            with self.director.stage(f'decomp_{scene}', duration=600) as sd:
                sd.fade_in(20).fade_out(10)
                grid = Animation.GridContainer(
                    padding=40,
                    text_line_height=140,
                    rows={ ' ': 1. },
                    cols={ 'Reference': 1., 'Ours': 1., 'R3DG': 1. },
                    align='left',
                )
                for name in ['R3DG', 'Ours', 'Reference']:
                    grid[' '][name] = Animation.Switch(
                        start=anim_switch['albedo'] - 10,
                        stop=anim_switch['albedo'] + 10,
                        before=Animation.ImageFrames(
                            folder=Path('..') / 'iccv' / 'decomp' / scene / name,
                            pattern="*.render.png",
                            use_rgba=False,
                            clip=clip,
                        ),
                        after=Animation.Switch(
                            start=anim_switch['relit1'] - 10,
                            stop=anim_switch['relit1'] + 10,
                            before=Animation.Switch(
                                start=99 + anim_desc['albedo'][0],
                                stop=100 + anim_desc['albedo'][0],
                                before=Animation.Highlight(
                                    Animation.ImageFrames(
                                        folder=Path('..') / 'iccv' / 'decomp' / scene / name,
                                        pattern="*.albedo.png",
                                        use_rgba=False,
                                        cut_out=anim_desc['albedo'][0],
                                        clip=clip,
                                    ),
                                    source=(anim_desc['albedo'][1], anim_desc['albedo'][2]),
                                    target=(anim_desc['albedo'][3], anim_desc['albedo'][4]),
                                    thickness=15,
                                    start=anim_desc['albedo'][0] + 10,
                                    stop=anim_desc['albedo'][0] + 90,
                                ),
                                after=Animation.ImageFrames(
                                    folder=Path('..') / 'iccv' / 'decomp' / scene / name,
                                    pattern="*.albedo.png",
                                    use_rgba=False,
                                    start=100,
                                    clip=clip,
                                ),
                            ),
                            after=Animation.Switch(
                                start=anim_switch['relit2'] - 10,
                                stop=anim_switch['relit2'] + 10,
                                before=Animation.Switch(
                                    start=anim_desc['relit1'][0] + 199,
                                    stop=anim_desc['relit1'][0] + 200,
                                    before=Animation.Highlight(
                                        Animation.ImageFrames(
                                            folder=Path('..') / 'iccv' / 'decomp' / scene / name,
                                            pattern="*.relit1.png",
                                            use_rgba=False,
                                            start=100,
                                            cut_out=anim_desc['relit1'][0],
                                            clip=clip,
                                        ),
                                        source=(anim_desc['relit1'][1], anim_desc['relit1'][2]),
                                        target=(anim_desc['relit1'][3], anim_desc['relit1'][4]),
                                        thickness=15,
                                        start=anim_desc['relit1'][0] + 110,
                                        stop=anim_desc['relit1'][0] + 190,
                                    ),
                                    after=Animation.ImageFrames(
                                        folder=Path('..') / 'iccv' / 'decomp' / scene / name,
                                        pattern="*.relit1.png",
                                        use_rgba=False,
                                        start=200,
                                        clip=clip,
                                    ),
                                ),
                                after=Animation.Switch(
                                    start=399,
                                    stop=400,
                                    before=Animation.ImageFrames(
                                        folder=Path('..') / 'iccv' / 'decomp' / scene / name,
                                        pattern="*.relit2.png",
                                        use_rgba=False,
                                        start=200,
                                        clip=clip,
                                    ),
                                    after=Animation.Switch(
                                        start=anim_desc['relit2'][0] + 299,
                                        stop=anim_desc['relit2'][0] + 300,
                                        before=Animation.Highlight(
                                            Animation.ImageFrames(
                                                folder=Path('..') / 'iccv' / 'decomp' / scene / name,
                                                pattern="*.relit2.png",
                                                use_rgba=False,
                                                start=400,
                                                cut_out=anim_desc['relit2'][0] - 200,
                                                clip=clip,
                                            ),
                                            source=(anim_desc['relit2'][1], anim_desc['relit2'][2]),
                                            target=(anim_desc['relit2'][3], anim_desc['relit2'][4]),
                                            thickness=15,
                                            start=anim_desc['relit2'][0] + 210,
                                            stop=anim_desc['relit2'][0] + 290,
                                        ),
                                        after=Animation.ImageFrames(
                                            folder=Path('..') / 'iccv' / 'decomp' / scene / name,
                                            pattern="*.relit2.png",
                                            use_rgba=False,
                                            start=500,
                                            clip=clip,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )

                sd[0.03:0.11, 0.05:] = Animation.StaticText(
                    f'Comparison ({scene_name})',
                    bold=True,
                    align='left',
                )
                sd[0.17:0.23, 0.05:0.18] = Animation.Switch(
                    start=anim_switch['albedo'] - 10,
                    stop=anim_switch['albedo'] + 10,
                    mode='push',
                    before=Animation.StaticText(
                        'Render',
                        bold=True,
                        align='left',
                    ),
                    after=Animation.Switch(
                        start=anim_switch['relit1'] - 10,
                        stop=anim_switch['relit1'] + 10,
                        mode='push',
                        before=Animation.StaticText(
                            'Albedo',
                            bold=True,
                            align='left',
                        ),
                        after=Animation.Switch(
                            start=anim_switch['relit2'] - 10,
                            stop=anim_switch['relit2'] + 10,
                            mode='push',
                            before=Animation.StaticText(
                                'Relit 1',
                                bold=True,
                                align='left',
                            ),
                            after=Animation.StaticText(
                                'Relit 2',
                                bold=True,
                                align='left',
                            ),
                        ),
                    ),
                )
                sd[0.25:0.95, 0.005:0.98] = grid

            with self.director.stage(f'decomp_{scene}_overview', duration=200) as sd:
                sd.fade_in(10).fade_out(20)
                grid = Animation.GridContainer(
                    padding=40,
                    text_line_height=90,
                    rows={ 'R3DG': 1., 'Ours': 1., 'Reference': 1. },
                    cols={ 'Render': 1., 'Albedo': 1., 'Envmap': 2., 'Relit 1': 1., 'Relit 2': 1. },
                    align='left',
                )

                for name in ['R3DG', 'Ours', 'Reference']:
                    for attr, col_name in zip(
                        ['render', 'albedo', 'relit1', 'relit2'],
                        ['Render', 'Albedo', 'Relit 1', 'Relit 2'],
                    ):
                        anim = Animation.ImageFrames(
                            folder=Path('..') / 'iccv' / 'decomp' / scene / name,
                            pattern=f"*.{attr}.png",
                            use_rgba=False,
                            clip=clip,
                        )
                        grid[name][col_name] = anim
                    grid[name]['Envmap'] = Animation.ImageFrames(
                        folder=Path('..') / 'iccv' / 'decomp' / scene / name,
                        pattern="envmap.png",
                        use_rgba=False,
                    )

                sd[0.03:0.11, 0.05:] = Animation.StaticText(
                    f'Overview ({scene_name})',
                    bold=True,
                    align='left',
                )
                sd[0.11:0.98, 0.05:0.95] = grid

        with self.director.stage('title_geometry', duration=60) as sd:
            sd.fade_in(20).fade_out(20)
            sd[0.4:0.6, 0.1:0.9] = Animation.StaticText(
                'Comparison on Normal Quality',
                bold=True,
            )

        with self.director.stage('normal_comparison', duration=200) as sd:
            sd.fade_in(20).fade_out(20)
            grid = Animation.GridContainer(
                padding=40,
                text_line_height=90,
                rows={ 'R3DG': 8., 'Ours': 8., 'Reference': 8. },
                cols={ 'Armadillo': 10., 'Hotdog': 10., 'Ball': 10., 'Teapot': 10., 'Helmet': 10. },
                align='left',
            )

            for attr, padding, clip in zip(
                ['arm', 'hotdog', 'ball', 'teapot', 'helmet'],
                [
                    (100, 0, 100, 0),
                    (100, 0, 100, 0),
                    (100, 0, 100, 0),
                    (100, 0, 100, 0),
                    (100, 0, 100, 0),
                ],
                [
                    (slice(40, -200), slice(150, -150)),
                    (slice(100, -60), slice(100, -100)),
                    None,
                    (slice(175, -225), slice(275, -225)),
                    (slice(80, -40), slice(75, -75)),
                ],
                strict=True,
            ):
                for name in ['R3DG', 'Ours', 'Reference']:
                    anim = Animation.ImageFrames(
                        folder=Path('..') / 'iccv' / 'normal' / attr / name,
                        use_rgba=(name == 'Ours' and attr in ('ball', 'helmet', 'teapot')) or (name == 'Reference'),
                        clip=clip,
                        padding_lbrt=padding,
                    )
                    grid[name][('armadillo' if attr == 'arm' else attr).capitalize()] = anim

            sd[0.03:0.11, 0.05:] = Animation.StaticText(
                'Comparison: Normal Quality',
                bold=True,
                align='left',
            )
            sd[0.11:0.98, 0.05:0.95] = grid

        with self.director.stage('title_more', duration=60) as sd:
            sd.fade_in(20).fade_out(20)
            sd[0.4:0.6, 0.1:0.9] = Animation.StaticText(
                'More Results of Our Method',
                bold=True,
            )

        with self.director.stage('shiny_decomp', duration=200) as sd:
            sd.fade_in(20).fade_out(20)
            grid = Animation.GridContainer(
                padding=40,
                text_line_height=90,
                rows={ 'Ball': 1., 'Teapot': 1., 'Helmet': 1. },
                cols={ 'Reference': 1., 'Render': 1., 'Albedo': 1., 'Roughness': 1., 'Envmap': 2. },
                align='left',
            )

            for name in ['ball', 'teapot', 'helmet']:
                for attr in ['reference', 'pbr', 'raw_albedo', 'roughness', '.']:
                    anim = Animation.ImageFrames(
                        folder=Path('outputs') / 'geosplat_s4' / name / 'dump' / attr,
                        use_rgba=attr != '.',
                        clip=(slice(100, -200), slice(175, -125)) if name == 'teapot' else None,
                    )
                    if attr == 'pbr':
                        attr = 'render'
                    elif attr == 'raw_albedo':
                        attr = 'albedo'
                    elif attr == '.':
                        attr = 'envmap'
                    grid[name.capitalize()][attr.capitalize()] = anim

            sd[0.03:0.11, 0.05:] = Animation.StaticText(
                'More Results: Ours on Reflective Cases',
                bold=True,
                align='left',
            )
            sd[0.11:0.98, 0.05:0.95] = grid

        with self.director.stage('more_relight', duration=220) as sd:
            sd.fade_in(20).fade_out(20)
            grid = Animation.GridContainer(
                padding=30,
                text_line_height=90,
                rows={ 'Hotdog': 1., 'Helmet': 1., 'Car': 1., 'Envmap': 0.5 },
                cols={ 'Bridge': 1., 'City': 1., 'Courtyard': 1., 'Fireplace': 1., 'Forest': 1., 'Night': 1., 'Sunrise': 1. },
                align='left',
            )

            for light in ['bridge', 'city', 'courtyard', 'fireplace', 'forest', 'night', 'sunrise']:
                for name in ['s4r_hotdog', 'helmet', 'sorb_car4']:
                    anim = Animation.Switch(
                        before=Animation.ImageFrames(
                            folder=Path('outputs') / 'geosplat_s4' / name / 'dump' / light,
                            use_rgba=False,
                            clip=(slice(None), slice(0, 800)),
                        ),
                        after=Animation.ImageFrames(
                            folder=Path('outputs') / 'geosplat_s4' / name / 'dump' / f'{light}_rr',
                            use_rgba=False,
                            clip=(slice(None), slice(0, 800)),
                            start=120,
                        ),
                        start=99,
                        stop=100,
                    )
                    if name == 's4r_hotdog':
                        name = 'hotdog'
                    elif name == 'sorb_car4':
                        name = 'car'
                    grid[name.capitalize()][light.capitalize()] = anim
                grid['Envmap'][light.capitalize()] = Animation.Switch(
                    before=Animation.ImageFrames(
                        folder=Path('outputs') / 'geosplat_s4' / 'helmet' / 'dump' / light,
                        use_rgba=False,
                        clip=(slice(None), slice(800, 2400)),
                    ),
                    after=Animation.ImageFrames(
                        folder=Path('outputs') / 'geosplat_s4' / 'helmet' / 'dump' / f'{light}_rr',
                        use_rgba=False,
                        clip=(slice(None), slice(800, 2400)),
                        start=120,
                    ),
                    start=99,
                    stop=100,
                )

            sd[0.03:0.11, 0.05:] = Animation.Switch(
                before=Animation.StaticText(
                    'More Relighting: Ours (Fixed Lighting)',
                    bold=True,
                    align='left',
                ),
                after=Animation.StaticText(
                    'More Relighting: Ours (Rotating Lighting)',
                    bold=True,
                    align='left',
                ),
                start=100,
                stop=120,
                mode='fade',
            )
            sd[0.11:0.98, 0.02:0.95] = grid

        with self.director.stage('title_thanks', duration=60) as sd:
            sd.fade_in(20).fade_out(20)
            sd[0.4:0.6, 0.1:0.9] = Animation.StaticText(
                'Thanks',
                bold=True,
            )

        if self.export:
            self.director.export(target_mb=self.target_mb)


if __name__ == '__main__':
    Script(cuda=0).run()
