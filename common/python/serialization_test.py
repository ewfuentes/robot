import unittest
from dataclasses import dataclass
from pathlib import Path
from common.python.serialization import dataclass_to_dict

@dataclass
class Inner:
    x: int
    y: str

@dataclass
class Outer:
    a: int
    b: Inner
    c: list
    d: dict
    e: Path

class TestDataclassToDict(unittest.TestCase):
    def test_simple_dataclass(self):
        @dataclass
        class Simple:
            foo: int
            bar: str
        obj = Simple(foo=1, bar="baz")
        expected = {"foo": 1, "bar": "baz"}
        self.assertEqual(dataclass_to_dict(obj), expected)

    def test_nested_dataclass(self):
        inner = Inner(x=10, y="test")
        outer = Outer(a=5, b=inner, c=[1, 2, 3], d={"k": 42}, e=Path("/tmp/file"))
        result = dataclass_to_dict(outer)
        self.assertEqual(result["a"], 5)
        self.assertEqual(result["b"], {"x": 10, "y": "test"})
        self.assertEqual(result["c"], [1, 2, 3])
        self.assertEqual(result["d"], {"k": 42})
        self.assertEqual(result["e"], "/tmp/file")

    def test_list_of_dataclasses(self):
        items = [Inner(x=i, y=str(i)) for i in range(3)]
        result = dataclass_to_dict(items)
        expected = [{"x": 0, "y": "0"}, {"x": 1, "y": "1"}, {"x": 2, "y": "2"}]
        self.assertEqual(result, expected)

    def test_dict_of_dataclasses(self):
        items = {"first": Inner(x=1, y="a"), "second": Inner(x=2, y="b")}
        result = dataclass_to_dict(items)
        expected = {"first": {"x": 1, "y": "a"}, "second": {"x": 2, "y": "b"}}
        self.assertEqual(result, expected)

    def test_none(self):
        self.assertIsNone(dataclass_to_dict(None))

    def test_path(self):
        p = Path("/home/user")
        self.assertEqual(dataclass_to_dict(p), "/home/user")

    def test_primitive(self):
        self.assertEqual(dataclass_to_dict(123), 123)
        self.assertEqual(dataclass_to_dict("abc"), "abc")
        self.assertEqual(dataclass_to_dict(3.14), 3.14)

if __name__ == "__main__":
    unittest.main()