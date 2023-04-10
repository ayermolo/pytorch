from typing import Optional

import torch._dynamo
from torch._dynamo.exc import UserError, UserErrorType
from torch.fx.experimental.symbolic_shapes import constrain_range


# TODO: we want to hide this min/max stuff under some abstraction similar to
# DynamicDim
def add_inline_constraint(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Add min/max constraint on the intermediate symbol at tracing time
    """

    try:
        constrain_range(symbol, min=min, max=max)
    except torch.utils._sympy.value_ranges.ValueRangeError as e:
        raise UserError(UserErrorType.CONSTRAIN_VIOLATION, e.args[0])

    return symbol

def add_inline_size_constraint(symbol, min: int = 2, max: Optional[int] = None):
    """
    Add min/max constraint on the intermediate symbol which will be used as a size
    """

    # TODO: we should investigate turning off 0/1 specialization for unbacked
    # SymInts
    if min < 2:
        raise UserError(
            UserErrorType.CONSTRAIN_VIOLATION,
            "Unable to set min size to be <= 2 because we specialize on 0/1 sizes."
        )
    return add_inline_constraint(symbol, min, max)
