# Practical example of time-dependent Neumann boundary conditions for a Fourier Neural Operator

### Introduction

The Fourier Neural Operator (FNO) has become a prominent architecture for learning mappings between function spaces directly from data. However, the literature presents differing statements regarding the scope of its implementation. In a key reference, the authors emphasize that:

"In order to maintain a fast and memory-efficient method, our implementation of the Fourier neural operator relies on the fast Fourier transform which is only defined on uniform mesh discretizations of , or for functions on the square satisfying homogeneous Dirichlet (fast Fourier sine transform) or homogeneous Neumann (fast Fourier cosine transform) boundary conditions" (Kovachki et al., 2023).

Following this reasoning, the publicly available NeuralOperator library (Kossaifi et al., 2025), which uses FFT-based convolutions, would also be formally limited when applied to non-homogeneous Neumann boundary conditions.

Conversely, another paper dedicated to FNO methodology states that FNOs are not restricted to homogeneous domains and can incorporate heterogeneous geometries and boundary conditions (Duruisseaux et al., 2025). This implies that, despite theoretical constraints of FFT-based solvers, FNOs may still perform adequately in practice when trained on non-homogeneous boundary conditions. An analogy may be drawn to the Nelder-Mead simplex algorithm, which lacks general convergence guarantees yet remains widely effective in applied settings.

To evaluate this in practice, I apply the FNO to a diffusion-reaction system with non-homogeneous boundary conditions and assess whether it can approximate the solution qualitatively.

### Theoretical setup

#### PDE model

For the model, we modify the 2D diffusion–reaction model from PDEBench (Takamoto et al., 2024). The model is made of two non-linearly coupled functions, the activator \(u = u(t,x,y)\) and the inhibitor \(v = v(t,x,y)\). The diffusion-reaction equations are:

$$
\begin{aligned}
\partial_t u &= D_u \partial_{xx} u + D_u \partial_{yy} u + R_u(u,v), \\
\partial_t v &= D_v \partial_{xx} v + D_v \partial_{yy} v + R_v(u,v),
\end{aligned}
$$

where \(D_u\) and \(D_v\) are diffusion coefficients for the activator and inhibitor, respectively, and \(R_u = R_u(u,v)\) and \(R_v = R_v(u,v)\) are the reaction functions.

The domain is:

$$
\Omega = [-1,1]^2, \quad (x,y)\in\Omega,\quad t\in(0,5].
$$

We choose explicitly the reaction functions:

$$
\begin{aligned}
R_u(u,v) &= u - u^3 - k - v + D_u \Delta u, \\
R_v(u,v) &= u - v + D_v \Delta v,
\end{aligned}
$$

where \(k = 5 \times 10^{-3}\), \(D_u = 1 \times 10^{-3}\), and \(D_v = 5 \times 10^{-3}\).

The system is subject to time-dependent non-homogeneous Neumann boundary conditions:

For \(u\):

$$
\begin{cases}
D_u \partial_x u = -0.05 t, & x = -1, \\
D_u \partial_x u = 0, & x = 1, \\
D_u \partial_y u = 0, & y = -1, \\
D_u \partial_y u = 0, & y = 1,
\end{cases}
$$

For \(v\):

$$
\begin{cases}
D_v \partial_x v = 0, & x = -1, \\
D_v \partial_x v = 0, & x = 1, \\
D_v \partial_y v = 0, & y = -1, \\
D_v \partial_y v = 0.1 \sin(2t), & y = 1.
\end{cases}
$$

These explicitly introduce spatially uneven and time-varying fluxes, departing from the standard homogeneous Neumann case. Previous tests with constant flux already gave satisfactory performance, motivating this more challenging configuration.

#### Neural operator

Neural operators aim to learn the solution operator:

$$
G : A \to U, \quad f(\cdot) \mapsto g(\cdot),
$$

where \(f(\cdot)\) is the input function, \(g(\cdot)\) is the output function and \(A,U\) are Banach spaces. In our case, we aim to learn:

$$
G' :
\begin{pmatrix}
u(x,y,t) \\
v(x,y,t)
\end{pmatrix}
\longrightarrow
\begin{pmatrix}
u(x,y,t + \Delta t) \\
v(x,y,t + \Delta t)
\end{pmatrix},
$$

using two channels to represent \(u\) and \(v\). We actually input 10 frames and output 10 frames to get the best results.

We recall the Universal Approximation Theorem for FNOs (Valentin Duruisseaux, 2025): Let

$$
G : H^s(\mathbb{T}^d; \mathbb{R}^{d_a}) \to H^{s'}(\mathbb{T}^d; \mathbb{R}^{d_u})
$$

be a continuous operator, and let \(K \subset H^s(\mathbb{T}^d; \mathbb{R}^{d_a})\) be compact. Then, for every \(\varepsilon > 0\), there exists a Fourier Neural Operator \(N\) such that:

$$
\sup_{a \in K} \|G(a) - N(a)\|_{H^{s'}} \le \varepsilon.
$$

This is well posed in our case.

#### Theoretical problem

We employ a Fourier Neural Operator (FNO), in which convolution in Fourier space is performed after a lifting step. The NeuralOperator library implements this using the Fast Fourier Transform (Kossaifi et al., 2025), which implicitly assumes periodicity outside the computational domain. Homogeneous Neumann boundary conditions would correspond to an even cosine expansion, requiring:

$$
\partial_n u = \partial_n v = 0 \quad \text{on } \partial\Omega
$$

where \(n\) is the normal vector to the boundary. However, in our setup, the imposed boundary fluxes are non-zero and spatially non-uniform, so these theoretical assumptions are violated. The goal is therefore to assess on a concrete example whether the FFT-based FNO can still learn an accurate solution operator under such conditions.


### Dataset

The dataset dataset contains down sampled versions of simulation, with specification and of the theoretical setup given previously. Here is an example of a random sample in the dataset:

![Figure 1](images/movie_2d_reacdiff.gif)

I generate 700 samples using the following modified rc_ode function from sim_diff_react.py in the PDEBench code:

&nbsp;   def rc_ode(self, t, y):  # noqa: ARG002

&nbsp;       """

&nbsp;       Solves a given equation for a particular time step.

&nbsp;       :param t: The current time step

&nbsp;       :param y: The equation values to solve

&nbsp;       :return: A finite volume solution

&nbsp;       """

&nbsp;       # Separate y into u and v

&nbsp;       u = y\[: self.Nx \* self.Ny\]

&nbsp;       v = y\[self.Nx \* self.Ny :\]

&nbsp;       # Calculate reaction function for each unknown

&nbsp;       react_u = u - u\*\*3 - self.k - v

&nbsp;       react_v = u - v

&nbsp;       # Boundary indices

&nbsp;       left_boundary = np.arange(0, self.Nx \* self.Ny, self.Nx)

&nbsp;       right_boundary = np.arange(self.Nx - 1, self.Nx \* self.Ny, self.Nx)

&nbsp;       top_boundary = np.arange(0, self.Nx)

&nbsp;       bottom_boundary = np.arange(self.Nx \* (self.Ny - 1), self.Nx \* self.Ny)

&nbsp;       # Neumann flux values

&nbsp;       g_u_left = -0.05 \* t \* np.ones_like(left_boundary)  

&nbsp;       g_u_right = 0 \* np.ones_like(right_boundary)

&nbsp;       f_v_bottom = 0 \* np.ones_like(bottom_boundary)

&nbsp;       f_v_top = 0.1 \* np.sin(2 \* t) \* np.ones_like(top_boundary)

&nbsp;       # Initialize boundary arrays (only boundary contribution)

&nbsp;       bc_u = np.zeros_like(u)

&nbsp;       bc_v = np.zeros_like(v)

&nbsp;       # Apply flux

&nbsp;       bc_u\[left_boundary\]  = g_u_left / self.dx

&nbsp;       bc_u\[right_boundary\] = g_u_right / self.dx

&nbsp;       bc_v\[bottom_boundary\]  = f_v_bottom / self.dy

&nbsp;       bc_v\[top_boundary\] = f_v_top / self.dy

&nbsp;       # Time derivatives

&nbsp;       u_t = react_u + self.Du \* (self.lap @ u) + bc_u

&nbsp;       v_t = react_v + self.Dv \* (self.lap @ v) + bc_v

&nbsp;       # Stack the time derivative into a single array y_t

&nbsp;       return np.concatenate((u_t, v_t))

### Training

I generated a sample of size 700, and trained a model on 500 samples, with 50 for the test. We then calculate the result on the 150 lefts. I used those specifications with the model:

initial_steps = 10

future_steps = 10

batch_size = 4

n_epochs = 50

operator = FNO(

&nbsp;   n_modes=(16,16,5),       # 3D Fourier modes: nx, ny, time

&nbsp;   hidden_channels=32,

&nbsp;   in_channels=dataset\[0\]\[0\].shape\[0\],  # number of input channels

&nbsp;   out_channels=dataset\[0\]\[1\].shape\[0\]  # number of output channels

)

### Results

On 150 new samples, unseen during the training phase, the FNO got

Evaluation on last 150 samples:

MSE : 0.000208

MAE : 0.008181

R² : 0.995152

This is rather very impressive, confirming that FNO are able to learn time-dependent, so non-homogeneous, Neumann boundary conditions, as shown by a concrete example below for and 10 frames as input and 10 frames as output:
![Figure 2](images/bc_0.png)
![Figure 3](images/bc_1.png)
We can also try on discretization invariance, and we can see that it still holds for the following example with the same system but with grid size :
![Figure 4](images/bc_test_2.png)

### Conclusion

There still seems like there is a blur as for the theoretical application of FNOs, in particular, whether FNO are compatible with PDEs with non-homogeneous Neumann boundary condition. Using the fast Fourier transform theoretically invalidates the use of non-homogeneous Dirichlet and Neumann boundary conditions. Yet, the current framework works fine even if the boundary conditions are non-homogeneous and even time dependent. This may be because of a wrong theoretical framework in the mathematical analysis part of neural operators.

Another possibility, as suggested in the introduction, is to be in a similar case as the Nelder Mead algorithm, where convergence is mathematically disproven, yet the practical use is really good.

### Bibliography

Jean Kossaifi, N. K.-S. (2025). _A Library for Learning Neural Operators._ Retrieved from arXiv:2412.10354

Makoto Takamoto, T. P. (2024). _PDEBENCH: An Extensive Benchmark for Scientific Machine Learning._ Retrieved from arXiv:2210.07182

Nikola Kovachki, Z. L. (2023). _Neural Operator: Learning Maps Between Function Spaces._ Retrieved from arXiv:2108.08481

Valentin Duruisseaux, J. K. (2025). _Fourier Neural Operators Explained: A Practical Perspective._ Retrieved from arXiv:2512.01421
