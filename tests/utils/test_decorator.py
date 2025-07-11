from rfstudio.utils.decorator import (
    batchify,
    chains,
    check_for_decorated_method,
    lazy,
    lazy_property,
)


def test_check_for_decorated_method():

    from functools import wraps

    def make_assert(a, b):
        assert a is b, (a, b)

    def d1(fn):
        @wraps(fn)
        def wrapper(self):
            check_for_decorated_method(self, fn, wrapper)
            return fn(self)
        return wrapper

    def d2(fn):
        def wrapper(self):
            check_for_decorated_method(self, fn, wrapper)
            return fn(self)
        return wrapper

    def d3(fn):
        wrapper_container = []
        wrapper_container.append(
            lambda self : (
                make_assert(check_for_decorated_method(self, fn, wrapper_container[0]), True)
                or
                fn(self)
            )
        )
        return wrapper_container[0]

    class A:

        @d1
        def f1(self):
            return 1

        @d2
        def f2(self):
            return 2

        @d3
        def f3(self):
            return 3

        @property
        @d1
        def p1(self):
            return 1

        @property
        @d2
        def p2(self):
            return 2

        @property
        @d3
        def p3(self):
            return 3


    a = A()
    assert a.f1() == 1
    assert a.f2() == 2
    assert a.f3() == 3
    assert a.p1 == 1
    assert a.p2 == 2
    assert a.p3 == 3


def test_lazy():

    class Test:

        def __init__(self):
            self.cnt = 0

        @lazy
        def plus(self, num: int = 1) -> int:
            self.cnt += num
            return self.cnt

        def plus_one(self) -> int:
            self.cnt += 1
            return self.cnt

        @lazy_property
        def count(self) -> int:
            return self.cnt

        @lazy(manually_decide=True)
        def to_string(self, fmt: str) -> int: # type: ignore
            yield fmt
            yield fmt.format(self.cnt)

    test = Test()
    assert test.cnt == 0
    assert test.to_string('{:03d}') == '000'
    assert test.plus() == 1 and test.cnt == 1
    assert test.count == 1
    assert test.plus_one() == 2 and test.cnt == 2 and test.count == 1
    assert test.plus(3) == 1 and test.cnt == 2 and test.count == 1
    assert test.to_string('{:03d}') == '000'
    assert test.to_string('{:04d}') == '0002'


def test_batchify():

    from dataclasses import dataclass

    import torch
    from torch import Tensor

    from rfstudio.utils.tensor_dataclass import Int, TensorDataclass

    class A:
        def __init__(self):
            self.batch_size = 3
            self.f2_batch_size = 5

        @batchify
        def f1(self, v):
            return v

        @batchify(prop='f2_batch_size')
        def f2(self, v):
            return torch.tensor(v.shape, dtype=torch.int32).view(1, -1)

    @dataclass
    class Data(TensorDataclass):
        key: Tensor = Int[..., 3]
        value: Tensor = Int[..., 2]
        info: Tensor = Int[2, 2]

    data = Data.zeros((17, 3))
    data.info[...] = torch.arange(4).view(2, 2)
    data.value[..., 0] = 1
    data.value[..., 1] = 2

    a = A()
    result = a.f1(data)
    assert (result.key == data.key).all()
    assert (result.value == data.value).all()
    assert (result.info == data.info).all()
    rand_data = torch.randn(18, 19)
    assert (rand_data == a.f1(rand_data)).all()
    result = a.f2(data)
    assert (result == torch.tensor([
        [5, 3],
        [5, 3],
        [5, 3],
        [2, 3]
    ], dtype=torch.int32)).all()


def test_chain():

    class A:

        @chains
        def add1(self, value: int):
            base = value
            def add2(self, value: int):
                return base + value
            return add2

    a = A()
    assert a.add1(3).add2(5) == 8


if __name__ == '__main__':
    test_check_for_decorated_method()
    test_lazy()
    test_batchify()
    test_chain()
