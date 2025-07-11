from dataclasses import dataclass

from rfstudio.engine.task import Task


@dataclass
class Test(Task):

    a: int = ...

    def run(self) -> None:
        """something"""
        print(1)
        assert 0

if __name__ == '__main__':

    test = Test()
    test.run()
