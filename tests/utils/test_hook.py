from typing import Callable

from rfstudio.utils.hook import enter_hook, exit_hook, inject, inject_once, wrap_hook


def test_inject():

    class A:
        def afoo(self) -> int:
            "adgkadlhgl"
            return 1

    class B(A):
        def bfoo(self) -> int:
            return 2

    def cfoo(self) -> int:
        return 3

    dfoo = lambda self : 4 # noqa: E731

    a = A()
    b = B()
    assert a.afoo() == 1
    assert b.bfoo() == 2
    inject(a.afoo, cfoo)
    inject(b.bfoo, dfoo)
    assert a.afoo() == 3
    assert b.bfoo() == 4
    inject(a.afoo, dfoo)
    assert a.afoo() == 4
    assert a.afoo.__doc__ == A.afoo.__doc__
    assert a.afoo.__qualname__ == A.afoo.__qualname__


def test_inject_once():

    class A:
        def afoo(self) -> int:
            "adgkadlhgl"
            return 1

    class B(A):
        def bfoo(self) -> int:
            return 2

    def cfoo(self) -> int:
        return 3

    dfoo = lambda self : 4 # noqa: E731

    a = A()
    b = B()
    assert a.afoo() == 1
    assert b.bfoo() == 2
    inject_once(a.afoo, cfoo)
    inject_once(b.bfoo, dfoo)
    assert a.afoo() == 3
    assert a.afoo() == 1
    assert b.bfoo() == 4
    assert b.bfoo() == 2
    inject_once(a.afoo, dfoo)
    assert a.afoo.__func__ is not A.afoo
    assert a.afoo() == 4
    assert a.afoo() == 1
    assert a.afoo.__func__ is A.afoo


def test_enter_hook():

    closure = [0]

    class A:
        def foo(self, s: int) -> None:
            closure[0] += s

    def hook(self, s: int) -> None:
        assert s == 2
        closure[0] += 1

    a = A()
    enter_hook(a.foo, hook)
    a.foo(2)
    assert closure[0] == 3


def test_exit_hook():

    closure = [0]

    class A:
        def foo(self) -> int:
            closure[0] += 1
            return closure[0] + 3

    def hook(self, r: int) -> None:
        assert closure[0] + 3 == r
        closure[0] = r

    a = A()
    exit_hook(a.foo, hook)
    a.foo()
    assert closure[0] == 4


def test_wrap_hook():

    closure = [0]

    class A:
        def foo(self) -> None:
            closure[0] += 1

    def hook(self, promise: Callable[[], None]) -> None:
        assert closure[0] == 0
        promise()
        assert closure[0] == 1
        closure[0] += 3

    a = A()
    wrap_hook(a.foo, hook)
    a.foo()
    assert closure[0] == 4


if __name__ == '__main__':
    test_inject()
    test_wrap_hook()
    test_enter_hook()
    test_exit_hook()
