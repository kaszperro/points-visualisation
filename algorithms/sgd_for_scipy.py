import numpy as np
from scipy.optimize import OptimizeResult


def sgd(
        fun,
        jac,
        x0,
        args=(),
        learning_rate=0.001,
        mass=0.9,
        startiter=0,
        maxiter=1000,
        callback=None,
        **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of stochastic
    gradient descent with momentum.
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    velocity = np.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        g = jac(x)

        if callback and callback(x):
            break

        velocity = mass * velocity - (1.0 - mass) * g
        x = x + learning_rate * velocity

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


def rmsprop(
        fun,
        jac,
        x0,
        args=(),
        learning_rate=0.1,
        gamma=0.9,
        eps=1e-8,
        startiter=0,
        maxiter=1000,
        callback=None,
        **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of root mean
    squared prop: See Adagrad paper for details.
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    avg_sq_grad = np.ones_like(x)

    for i in range(startiter, startiter + maxiter):
        print(fun(x, *args), i)
        if i > 800:
            learning_rate = 0.01
        g = jac(x,  *args)

        if callback and callback(x):
            break

        avg_sq_grad = avg_sq_grad * gamma + g ** 2 * (1 - gamma)
        x = x - learning_rate * g / (np.sqrt(avg_sq_grad) + eps)

    return x


def adam(
        fun,
        jac,
        x0,
        args=(),
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        startiter=0,
        maxiter=1000,
        callback=None,
        **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    best_energy = fun(x, *args), x
    for i in range(startiter, startiter + maxiter):
        energy =fun(x, *args)
        print(energy)
        if energy < best_energy[0]:
            best_energy = energy, np.copy(x)

        if energy < 4459.1:
            return best_energy[1]
        g = jac(x, *args)

        if callback and callback(x):
            break

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g ** 2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1 ** (i + 1))  # bias correction.
        vhat = v / (1 - beta2 ** (i + 1))
        x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

        if i == (startiter + maxiter) // 2:
            learning_rate /= 2






    return best_energy[1]
