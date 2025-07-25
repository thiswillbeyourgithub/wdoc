"""
In Langchain, using the @debug decorator turns callable into runnable. But
if something created by `def` in python is not a callable, beartype crashes.

Hence, the CallableRunnableLambda class simply add a dummy __call__ method to
allow package-wide type checking.

Original source code:
https://github.com/langchain-ai/langchain/blob/0d6f9154420652ccd61cf3ff6e86393816d7520f/libs/core/langchain_core/runnables/base.py#L5993
"""

from langchain_core.runnables.base import RunnableLambda
from beartype.typing import Callable, Union


class CallableRunnableLambda(RunnableLambda):
    """RunnableLambda with __call__ method for beartype compatibility."""

    def __call__(self, *args, **kwargs):
        raise Exception("CallableRunnableLambda should never be called")


# can be used as decorator on top of functions or directly on top of RunnableLambda
def callable_chain(input: Union[Callable, RunnableLambda]) -> CallableRunnableLambda:
    "Like @chain but also a callable, for beartype compatibility"
    if isinstance(input, CallableRunnableLambda):
        # should never happen
        raise TypeError(input)
    elif isinstance(input, RunnableLambda):
        assert not callable(input)
        assert hasattr(input, "func")
        assert callable(input.func)
        out = CallableRunnableLambda(input.func)
    elif callable(input):
        out = CallableRunnableLambda(func)
    else:
        raise TypeError(input)

    assert isinstance(out, CallableRunnableLambda)
    assert isinstance(out, RunnableLambda)
    assert callable(out)
    return out
