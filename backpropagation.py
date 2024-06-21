"""
perform backpropagation to compute gradients

implements a simple automatic differentiation system for scalar values.

features:
1. Value objects can be combined using arithmetic operations.
2. The backward method computes gradients using reverse-mode automatic differentiation.
3. A topological sort is used to ensure correct order of gradient computation.

"""

from __future__ import annotations
from typing import Union, Callable, List


class Value:
    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: Union[Value, float]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other: Union[Value, float]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]) -> Value:
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Union[Value, float]) -> Value:
        return self + (-other)

    def __truediv__(self, other: Union[Value, float]) -> Value:
        return self * other**-1

    def __rtruediv__(self, other: Union[Value, float]) -> Value:
        return other * self**-1

    def relu(self) -> Value:
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self) -> None:
        topo: List[Value] = []
        visited = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()


def main():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c.relu()
    d.backward()

    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c: {c}")
    print(f"d: {d}")


if __name__ == "__main__":
    main()
