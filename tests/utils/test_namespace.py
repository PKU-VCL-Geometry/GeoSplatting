from rfstudio.utils.namespace import Namespace

if __name__ == '__main__':

    class test1(Namespace):

        a: int = 3

        b: bool = 4

        @staticmethod
        def foo(i: int) -> float:
            return i * 1.2

    assert test1.a == 3
    assert test1.b == 4
    assert test1.foo(2) == 2.4

    msg = None
    try:
        class test2(Namespace):
            __some = 3
    except AssertionError as e:
        msg = e.args
    assert msg == ('Private member is not allowed in Namespace. (`some`)',), msg

    msg = None
    try:
        class test3(Namespace):
            def __init__(self):
                pass
    except AssertionError as e:
        msg = e.args
    assert msg == ('Magic is not allowed in Namespace. (`__init__`)',), msg

    msg = None
    try:
        class test4(Namespace):
            def foo(i: int) -> float:
                return i * 1.2
    except AssertionError as e:
        msg = e.args
    assert msg == ('Only staticmethod is allowed in Namespace. (`foo`)',), msg
