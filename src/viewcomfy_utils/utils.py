from typing import Any


class AlwaysEqualProxy(str):
    __slots__ = ()

    def __eq__(self, obj: object) -> bool:
        return True

    def __ne__(self, obj: object) -> bool:
        return False


COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
    "a and b": lambda a, b: a and b,
    "a or b": lambda a, b: a or b,
}


class TautologyStr(str):
    __slots__ = ()

    def __ne__(self, other) -> bool:
        return False


class ByPassTypeTuple(tuple):
    __slots__ = ()

    def __getitem__(self, index: int) -> TautologyStr | Any:
        index = min(index, 0)
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item
