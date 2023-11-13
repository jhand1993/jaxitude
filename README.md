# JAXitude: Intuitive Attitude Calculations, Conversions, and State Estimation Modeling with JAX

### Why JAXitude?

JAXitude started out as a project with which to become more familear with JAX while providing a useful tool for calculating attitude transformations and compositions.
While still providing these base calculations, JAXitude also provides tools for rapid prototyping of rigid body state estimation algorithms and realistic kinematics simulations.
All of these tools and workflows are JAX-compatible, meaning you can exploit the autodifferentiation and acceeleration functionality of JAX in your own JAXitude projects.

### What is JAXitude For?

As mentioned, JAXitude is useful for algorithm protoyping and iteration, along with trajectory simulations for rigid bodies.
Although JAX provides some impress ways to dramatically speed up JAXitude workflows, JAXitude is not really meant for production state estimation algorithm implementations.

### Roadmap

JAXitude v0.4.0 plans to add IMU and magnetometer noise modeling along with kinematic equation integrators to simulate simple state trajectories and corresponding measurements.

JAXitude v0.5.0 plans to add functionality for building and simulating SLAM algorithms using filters and Bayes nets.

JAXitude v0.6.0 plans to add functionality for building and simulating optimization-based SLAM algorithms such as factor graphs.

## Installation

JAXitude can be installed directly from github or via pip (conda installation will be added at a later date):

```bash
pip install jaxitude
```

JAXitude only requies jax (at least v0.4.16), jaxlib (at least v0.4.14), and scipy ( at least v1.11.3).
matplotlib is required to run notebooks provided under `demos/`.

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

Why require this?
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

JAXitude provides multiple attitude representation formalism for you to utilize, along with methods to convert between them.
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

`MRP` will automatically calculate the DCM (rotation matrix) corresponding to $\sigma$, which can be accessed by calling the object `mrp_obj`.

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

### Composition Operations

Each representation of rigid body orientations (rotations) are are representations of the $\text{SO}(3)$ group.
Because rotations form a group, you cannot simply add two rotation and get another rotation.
Instead, we have to compose rotations, and this composition operation depends on how we choose to represent rotations.

Let's define two rotations represented with quaternions $\beta_1$ and $\beta_2$.

```Python
import jax.numpy as jnp

# pi/2 rotation along the x-axis.
b1 = jnp.array(
    [[jnp.sqrt(2.) / 2.],
     [jnp.sqrt(2.) / 2.],
     [0.],
     [0.]]
)
# pi/2 rotation along the y-axis.
b2 = jnp.array(
    [[jnp.sqrt(2.) / 2.],
     [0.],
     [jnp.sqrt(2.) / 2.],
     [0.]]
)
```

We can use the `jaxitude.quaternions.Quaterion` object to instantiate quaternion objects for calculations.

```Python
from jaxitude.quaternions import Quaternion

quat1 = Quaternion(b1)
quat2 = Quaternion(b2)
```

From these `Quaternion` instances, we can grab the two corresponding rotation matrices by calling each object instance.
These rotation matrices (or DCMs) can then be composed using matrix multiplication to get the combination of rotations 1 and 2.

```Python
R3 = quat2() @ quat2()  # equivalent to R1 @ R2.
```

Alternatively, we could use the quaternion composition operation to directly compose as provided by the submodule `jaxitude.operations.composition`.

```Python
from jaxitude.operations.composition import compose_quat

b3 = compose_quat(b1, b2)
```

`jaxitude.operations.composition` provides the composition operations for CRPs $q$ and MRPs $\sigma$ as well.

```Python
from jaxitude.operations.composition import compose_crp, compose_mrp

# Equivalent CRP rotations.
q1 = jnp.array(
    [[1.],
     [0.],
     [0.]]
)
q2 = jnp.array(
    [[0.],
     [1.],
     [0.]]
)

# Equivalent MRP rotations.
s_mag = jnp.sqrt(2.) / 2. / (1. + jnp.sqrt(2.) / 2.)
s1 = jnp.array(
    [[s_mag],
     [0.],
     [0.]]
)
s2 = jnp.array(
    [[0.],
     [s_mag],
     [0.]]
)

q3 = compose_crp(q1, q2)
s3 = compose_mrp(s1, s2)
```

### Attitude Rates and Integrating Rate Equations

Unlike attitudes, which in being rotations are group elements of $\text{SO}(3)$, attitude rates (angular velocities) $\mathbf{\omega}$ *are* vectors (Lie algebras for the win).
Rigid body kinematics, which provide the mathematical foundation of trajectory simulations and rigid body control laws, are most elegantly written in terms of $\mathbf{\omega}$ -- I.e. Euler's equations of motion.
Moreover, sensors such as gyroscopes directly output measured $\mathbf{\omega}$.
In state estimation and trajectory simulations, we need to be able to transform $\mathbf{\omega}$ values to corresponding quaternion, CRP, and MRP rates ($\dot{\beta}$, $\dot{q}$, and $\dot{\sigma}$, respectively).
JAXitude provides this functionality in the `jaxitude.operations.evolution` submodule.

Let us simulate attitude rates using the following function.

```Python
import jax.numpy as jnp

# w_t is a function that returns an attitude rate vector given a time t.
w_t = lambda t: jnp.array(
    [[jnp.cos(0.5 * jnp.pi * t)],
     [-jnp.cos(0.25 * jnp.pi * t)],
     [jnp.sin(0.5 * jnp.pi * t)]]
)
```

Our objective will to calculate $\beta(t)$ along the interval $t=[0, 30]$ by integrating $\dot{\beta}(t)$.
To do so, we'll import `jaxitude.operations.evolution.evolve_quat()` and wrap it with a helper function `dbdt(t)`.

```Python
from jax import jit

from jaxitude.operations.evolution import evolve_quat

# Jitted wrapper function to yield dbdt at time t.
# Note that the first argument for differential equations should be the time argument in JAXitude.
@jit
def dbdt(
    t: float,
    b: jnp.ndarray
) -> jnp.ndarray:
    return evolve_quat(b, w_t(t))
```

Now we can use one of the differential equation solvers (also called integrators here) provided with `jaxitude.operations.integrator`.
Specifically, we'll utilize the Runge-Kutta 4 integrator `rk4()`.
We also need to provide the initial attitude $\beta_0 = \beta(t=0)$ to begine solving.

IMPORTANT NOTE: JAXitude provides integrators that are for autonomous differential equations (no explicit time dependece) and nonautonomous differential equations (explicit time dependence).
In this example, `dbdt()` is explicitly dependent on time, hence why we are using `rk4()` in place of `autonomous_rk4()`.
Also, when defining a function representing a nonautonomous differential equation to pass through JAXitude nonautonomous integrator, the first argument said function must be the time argument, as is the case for our definition of `dbdt(t, b)`.

```Python
from jaxitude.operations.integrator import rk4

# b(t=0) will be a zero rotation.
b = jnp.array(
    [[1.],
     [0.],
     [0.],
     [0.]]
)

# Initialize list of b values.
b_list = []

# Solve from t=0 to t=30 in 500 equal steps.
t_arr = jnp.linspace(0., 30., 500)
dt = t_arr[1] - t_arr[0]

# Loop through and solve for b(t).
for t in t_arr:
    b_list.append(b)
    b = rk4(
        dbdt,
        t,
        b,
        dt
    )

b_arr = jnp.array(b_list)
```

Similar attitude rate differential equations exist for $\dot{q}$ (`evolution.evolve_crp()`) and $\dot{\sigma}$ (`evolution.evolve_mrp()`). 
Currently, JAXitude does not implement differential equations relating $\mathbf{\omega}$ to Euler angle rates ($\dot{\alpha}$, $\dot{\beta}$, $\dot{\gamma}$).
