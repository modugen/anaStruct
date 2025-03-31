import operator

import numpy as np
from numpy.core import numeric as _nx
import numpy as np
import types

from numpy.core import numeric as _nx, ndim

@array_function_dispatch(_linspace_dispatcher)
def fast_linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
                  axis=0):
    num = operator.index(num)
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)
    div = (num - 1) if endpoint else num

    # Convert float/complex array scalars to float, gh-3504
    # and make sure one can use variables that have an __array_interface__, gh-6634
    # start = asanyarray(start) * 1.0
    # stop  = asanyarray(stop)  * 1.0

    dt = np.result_type(start, stop, float(num))
    if dtype is None:
        dtype = dt

    delta = stop - start
    y = _nx.arange(0, num, dtype=dt).reshape((-1,) + (1,) * ndim(delta))
    # In-place multiplication y *= delta/div is faster, but prevents the multiplicant
    # from overriding what class is produced, and thus prevents, e.g. use of Quantities,
    # see gh-7142. Hence, we multiply in place only for standard scalar types.
    _mult_inplace = _nx.isscalar(delta)
    if div > 0:
        step = delta / div
        if _nx.any(step == 0):
            # Special handling for denormal numbers, gh-5437
            y /= div
            if _mult_inplace:
                y *= delta
            else:
                y = y * delta
        else:
            if _mult_inplace:
                y *= step
            else:
                y = y * step
    else:
        # sequences with 0 items or 1 item with endpoint=True (i.e. div <= 0)
        # have an undefined step
        step = np.NaN
        # Multiply with delta to allow possible override of output class.
        y = y * delta

    y += start

    if endpoint and num > 1:
        y[-1] = stop

    if axis != 0:
        y = _nx.moveaxis(y, 0, axis)

    if retstep:
        return y.astype(dtype, copy=False), step
    else:
        return y.astype(dtype, copy=False)
