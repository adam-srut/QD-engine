# Quantum Dynamics engine

Exact quantum dynamics on one- or two-dimensional potentials from *ab initio* calculations. 

## Theory
### Hamiltonian
Only Hamiltonians in the following form are supported:

```math
\begin{equation}
\hat{\mathcal{H}} = -\frac{\hbar^2}{2\mu_x}\frac{\partial^2}{\partial x^2} -\frac{\hbar^2}{2\mu_y}\frac{\partial^2}{\partial y^2} + \hat{\mathcal{V}}(x,y)
\end{equation}
```

*i.e.* dimensions $x$ and $y$ need to be linear motions in a many-dimensional space of atomic nuclei.
### Propagator
The split operator formalism considers the non-commutability of $\hat{T}$ and $\hat{V}$, see the equation below. In order to easily apply the exponential form of a semi-local operator $\hat{T}$ the wavefunction is Fourier transformed to the momentum space where $\hat{T}$ is a local operator.

```math
\begin{equation}
\psi(x,y,t+\Delta t) = e^{-\frac{i}{\hbar}\frac{\hat{\mathcal{T}}}{2}\Delta t}
    e^{-\frac{i}{\hbar}\frac{\hat{\mathcal{V}}}{2}\Delta t}
    e^{-\frac{i}{\hbar}\frac{\hat{\mathcal{V}}}{2}\Delta t}
    e^{-\frac{i}{\hbar}\frac{\hat{\mathcal{T}}}{2}\Delta t}
    \psi(x,y,t)
\end{equation}
```

## Implementation
The time propagation in the package is implemented with 3 operators:

```math
\begin{align}
\hat{\mathcal{P}}_\mathrm{step} &=
    e^{-\frac{i}{\hbar}\hat{\mathcal{T}}\Delta t}
    e^{-\frac{i}{\hbar}\hat{\mathcal{V}}\Delta t}\\
\hat{\mathcal{P}}_\mathrm{init} &= 
    e^{-\frac{i}{\hbar}\frac{\hat{\mathcal{T}}}{2}\Delta t}\\
\hat{\mathcal{P}}_\mathrm{end} &= 
    e^{-\frac{i}{\hbar}\frac{\hat{\mathcal{T}}}{2}\Delta t}e^{-\frac{i}{\hbar}\hat{\mathcal{V}}\Delta t}
\end{align}
```
it is then only possible to propagate the wavefunction by $N+1$ steps with:
```math
\begin{equation}
\psi(x,y,t+\Delta t(N+1)) = \hat{\mathcal{P}}_\mathrm{end} \left( \hat{\mathcal{P}}_\mathrm{step} \right)_N \hat{\mathcal{P}}_\mathrm{init}~\psi(x,y,t)
\end{equation}
```
Such implementation allows to omit Fourier transform two times per step and thus enhances performance. 
Note that in order to compute the correlation function it is necessary to end the propagation to the next integer time step.

**More descriptive documentation of the package usage is coming soon.**
