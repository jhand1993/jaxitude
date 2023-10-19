# JAXitude

Intuitive attitude calculations and estimation written with JAX.

## Installation

JAXitude can be installed directly from github or via pip:

```bash
pip install jaxitude
```

JAXitude only requies jax (at least v0.4.16), jaxlib (at least v0.4.14), and scipy ( at least v1.11.3).

## JAXitude Fundamentals

JAXitude tries its best to be functional and consistent with JAX.
As such, it is good practice to not be lazy with typing and to try one's best to use JAX arrays.
All JAXitude functions and methods aim to be JAX-compliant, meaning if you expect the function to be jittable or vmappable, then it should be so.  
Custom JAXitude workflows are also expected to be functional, meaning it should flow not unlike if you were to be writing out the algorithm or mathematics by hand --- see `demos/` for some examples!

JAXitude uses the following notation and conventions.

- Unit quaternions are denoted by $\beta$ (`b` in the source code), with the first component $\beta_0$ being the scalar part of the quaternion.

- Classical Rodrigues parameters (CRPs) are denoted by $q$ (`q` in the source code).

- Modified Rodrigues parameters (MRPs) are denoted by $\sigma$ (`s` in the source code).

- Euler angles are denoted with $(\alpha, \beta, \gamma)$ (`alpha`, `beta`, and `gamma` in the source code), with $\alpha$ being the first rotation angle, $\beta$ the second, and $\gamma$ the third. The Euler angle types are provided as strings. For example, a classical Euler angle Z-X-Z rotation is denoted as either `'zxz'` or `'131`.

- Directional Cosine Matrices (DCMs), used interchangeably with rotation matrices throughout, represent passive rotations, not active rotations. This means a DCM transforms the underlying coordinate basis during a rotation --- it does not actively rotate vectors.

- All mathematical matrices and vectors are and should be represented with the appropriately shaped two-dimensional JAX arrays.

### The Ultimate Rule of JAXitude

The Ultimate Rule is that any object corresponding to a mathematical vector of dimension $N$ *must be* a $N\times1$ JAX array, unless otherwise stated.
As an example, say you want to represent a set of attitude rates $\mathbf{\omega}=[\omega_1, \omega_2, \omega_3]^T$ in a Python script (denoted `w` in the code below).
Here are some options.

```Python
import numpy as np
import jax.numpy as jnp

w = [w1, w2, w3]  # Use a list.
w = np.array([w1, w2, w3])  # Use 1D numpy array.
w = jnp.array([w1, w2, w3])  # Use a 1D jax.numpy array.
```

All of these options will not work in JAXitude.
Instead, $\mathbf{\omega}$ should be represented using a 2D jax array.

```Python
w = jnp.array(
    [[w1],
     [w2],
     [w3]]
)
```

Why requires this?
For starters, it forces the user to write code that represents mathematical vectors more accurately.
For JAXitude, it also provides an API that lends naturally better representing the underlying mathematics of vector algebra, calculus, and dynamical systems.
It also ostensibly helps with source code readibility.

Explicitly representing column vectors as 2D arrays requires changes to a numpy-like workflow.
For example, these functions work just fine with an explicit column vector.

```Python
w = jnp.array(
    [[w1],
     [w2],
     [w3]]
)

jnp.vdot(w, w)  # jnp.dot(w, w) won't work, though!
jnp.dot(w.T, w)  # This works, but returns a 2D object.
jnp.linalg.norm(w)
jnp.cos(w)
jnp.log(w)
```

The two sticking points are `jnp.dot()` (and `jnp.matmul()` by proxy) and `jnp.cross()`.

```Python
jnp.dot(w, w)  # Fails.
jnp.cross(w, w)  # Fails.
```

`jnp.vdot()` is preferred over `jnp.dot()` for this reason.  
As for calculating the cross product, the module `jaxitude.base` comes with a handy function `colvec_cross()` to overcome is issue.

```Python
from jaxitude.base import colvec_cross

colvec_cross(w, w)  # Success!
```

One final detail about the Ulimate Rule --- even Euler angle triples $(\alpha, \beta, \gamma)$, MRPs $q$, and CRPs $\sigma$ have to represented with 2D jax arrays, even though these are objects are not elements of mathematical vectors spaces.

## Attitude Representations and Conversions

JAXitude comes with multiple attitude representation formalism types for you to utilize, along with methods to convert between them.
These types are not usually used directly in calculation workflows (like in filter or control algorithms, for example).
Instead, they provide a consistent API with JAXitude to swap representations as needed.

Say we want to build an `MRP` object from $\sigma=(0.1, 0.2, 0.3)$ from the module `jaxitude.rodrigues`.

```Python
from jaxitude.rodrigues import MRP

s = jnp.array(
    [[0.1],
     [0.2],
     [0.3]]
)

mrp_obj = MRP(s)
```

`MRP` will automatically calculate the DCM corresponding to $\sigma$, which can be accessed by calling the object `mrp_obj`.

```Python
dcm = mrp_obj()  # dcm is a 3x3 jax array.
```

Now say we want to get the quaternions $\beta$ that correspond to $\sigma$.
We actually have two options.
The first is to calculate $\beta$ from the `mrp_obj.dcm` attribute directly by calling either the methods `mrp_obj.get_b_short()` or `mrp_obj.get_b_long()`.

```Python
b_short = mrp_obj.get_b_short()  # shorter rotation angle
b_long = mrp_obj.get_b_long()  # longer rotation angle
```

Any attitude object, regardless of formalism, has access to these two methods presented above.

Alternatively, we could calculate $\beta$ directly from $\sigma$.
Since `mrp_obj` is an instance of `jaxitude.rodrigues.MRP`, we could use the method `mrp_obj.get_b_from_s`.

```Python
b_from_s = mrp_obj.get_b_from_s()
```

All three methods return valid quaternions that correspond to $\sigma$, with `b_short == b_from_s` in most cases.
But in this case, `mrp_obj.get_b_from_s()` is $\approx 5\times$ faster than `mrp_obj.get_b_short()`.
In general, the methods `get_*_from_*()` will be slightly faster than their corresponding `get_*()` alternatives.

Sticking with the MRP example, let us now get the X-Y-Z Euler angle triplet corresponding to $\sigma$ to see how we get Euler angle representations.

```Python
euler_angle_type = '123'  # equivalent to 'xyz'
euler_angles = mrp_obj.get_eulerangles(euler_angle_type)
```

With Euler angles, there will not be specific methods converting from a given representation (such as in this case, MRPs) to Euler angles.
Also, you need to specify via a string the Euler angle type.
Otherwise, the workflow is the same as the quaternion transformation above!
