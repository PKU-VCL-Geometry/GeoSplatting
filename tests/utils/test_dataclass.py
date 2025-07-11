from dataclasses import dataclass
from pathlib import Path

from rfstudio.data import MultiViewDataset
from rfstudio.utils.dataclass import dump_dataclass_as_str, load_dataclass_from_str


@dataclass
class DataClass:

    a: int

    b: str

    c: MultiViewDataset

    d: float = 3.2


if __name__ == '__main__':

    data1 = MultiViewDataset(path=Path('.') / 'data')
    data2 = DataClass(a=1, b='st', c=data1)

    dump1 = dump_dataclass_as_str(data1)
    dump2 = dump_dataclass_as_str(data2)

    assert dump1 == dump_dataclass_as_str(load_dataclass_from_str(dump1))
    assert dump2 == dump_dataclass_as_str(load_dataclass_from_str(dump2))
