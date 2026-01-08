# Blurred predictions in Fourier Neural Operators with odd grids
### Introduction
I discovered neural operators a few weeks ago, and their discretization invariance immediately stood out as a key feature. While experimenting with a diffusion-reaction PDE, I observed that inference could fail for certain grid sizes, producing a blurred output. In this short note, I present a hypothesis to explain this phenomenon. I begin with my initial hypothesis and will explore further as time allows alongside my classes.

### Setup
Using the Python library NeuralOperator, I trained a Fourier Neural Operator (FNO) with the following configuration:
- operator = FNO(
    - n_modes=(16, 16, 5),
    - hidden_channels=32,
    - in_channels=2,
    - out_channels=2
)

The model was trained on the diffusion-reaction dataset from PDEBench (Takamoto et al., 2024). Due to hardware constraints, training was performed for 50 epochs using 500 training samples and 50 test samples. Each sample consists of 100 timesteps and 2 channels, representing an activator and an inhibitor.

### Problem
I then performed predictions on new samples simulated using the PDEBench code. The results for grid sizes 128×30, 128×31, 128×32, and 128×33 are as follows:
 
 
 
 
It appears that the FNO’s inference can be completely disrupted by the grid size. In particular, an odd-sized axis seems to destroy the predictions entirely, even for otherwise high-resolution grids. Interestingly, the borders along the even-sized axis remain largely unaffected, and it does not seem to matter whether the odd dimension is along the x- or y-axis.
 
### Possible explanation
My first hypothesis focuses on the Fast Fourier Transform (FFT).

On a finite grid

$$
u[i,j], \quad i = 0,\dots,N_x-1, \quad j = 0,\dots,N_y-1
$$

the Fourier layer computes the discrete Fourier transform for each coordinate:

$$
\hat{u}[i,j] = \sum_{k=0}^{N_x-1} \sum_{l=0}^{N_y-1} u[k,l] \, e^{2\pi i \left( \frac{ki}{N_x} + \frac{lj}{N_y} \right)}
$$

Recalling that \( e^{ix} = \cos x + i \sin x \), we have:

$$
e^{2\pi i \left( \frac{ki}{N_x} + \frac{lj}{N_y} \right)} = \cos\Big(2\pi \big(\frac{ki}{N_x} + \frac{lj}{N_y}\big)\Big) + i \sin\Big(2\pi \big(\frac{ki}{N_x} + \frac{lj}{N_y}\big)\Big)
$$

For a grid size 128x31, Ny=31 is prime. This means that most combinations of lj/Ny do not align with the discrete frequencies, except at the boundaries. Indeed, for small l, we have:

$$
\frac{lj}{N_y} \ll \frac{ki}{N_x} \quad \implies \quad \cos\Big(2\pi \big(\frac{ki}{N_x} + \frac{lj}{N_y}\big)\Big) \approx \cos\Big(2\pi \frac{ki}{N_x}\Big)
$$

which behaves as expected. When l approaches 31, lj/Ny aligns almost with the discrete frequencies.  

Interestingly, when using the Fourier layer code from the NeuralOperator theory folder (Kossaifi et al., 2025), this effect is not observed. In that code, random weights are applied iteratively through the Fourier layer. For a 10-timestep prediction run successively 32 times, no structured pattern emerges, and no interference is seen.


 
This suggests the following mechanism: with random weights, no structure is encoded, so the grid irregularity does not “destroy” anything. With a trained FNO, however, the neurons rely on precise frequency information. If the FFT produces slightly misaligned coefficients (as with an odd or prime axis size), the network receives incorrect inputs, which propagates through the layers, producing a completely blurred or nonsensical output in the interior.
### Conclusion
Predictions using a trained FNO are not fully “discretization invariant” in practice. While increasing the grid size generally works, certain sizes, particularly odd or prime axes, can lead to completely incorrect predictions. In our example, these grid sizes caused the FNO to fail entirely. A plausible explanation is that the Fast Fourier Transform on an odd-sized grid produces misaligned frequency components.
### Bibliography
Jean Kossaifi, N. K.-S. (2025). A Library for Learning Neural Operators. Retrieved from arXiv:2412.10354
Makoto Takamoto, T. P. (2024). PDEBENCH: An Extensive Benchmark for Scientific Machine Learning. Retrieved from arXiv:2210.07182

