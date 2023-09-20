# Quantum Dynamics engine

Exact quantum dynamics on one- or two-dimensional potentials from *ab initio* calculations. 

## Installation
The package relies only on Julia (tested against version 1.9.0) and does not have to be compiled. The following packages are needed:
```
ArgParse, Plots, NCDatasets, Trapz, SpecialPolynomials, FFTW, OffsetArrays, YAML, Dates, UnicodePlots, Interpolations
```
To enhance performance you can use Intel's MKL for Fourier transform with `FFTW.set_provider!("mkl")` in Julia REPL.

## Theory
### Hamiltonian
Only Hamiltonians in the following form are supported:

```math
\begin{equation}
\hat{\mathcal{H}} = -\frac{\hbar^2}{2\mu_x}\frac{\partial^2}{\partial x^2} -\frac{\hbar^2}{2\mu_y}\frac{\partial^2}{\partial y^2} + \hat{\mathcal{V}}(x,y)
\end{equation}
```

*i.e.* dimensions $x$ and $y$ need to be mutually orthogonal, linear motions in a many-dimensional space of atomic nuclei. Propagation of the wave packet is then done adiabatically on the potential $\hat{\mathcal{V}}(x,y)$.

### Propagator
The split operator formalism [1] considers the non-commutability of $\hat{T}$ and $\hat{V}$, see the equation below. In order to easily apply the exponential form of a semi-local operator $\hat{T}$ the wavefunction is converted to the momentum space via Fourier transform where $\hat{T}$ is a local operator.

```math
\begin{equation}
\psi(x,y,t+\Delta t) = e^{-\frac{i}{\hbar}\frac{\hat{\mathcal{T}}}{2}\Delta t}
    e^{-\frac{i}{\hbar}\frac{\hat{\mathcal{V}}}{2}\Delta t}
    e^{-\frac{i}{\hbar}\frac{\hat{\mathcal{V}}}{2}\Delta t}
    e^{-\frac{i}{\hbar}\frac{\hat{\mathcal{T}}}{2}\Delta t}
    \psi(x,y,t)
\end{equation}
```

---
&nbsp;&nbsp;&nbsp;&nbsp;[1] Feit, Fleck & Steiger, *J. Comput. Phys.*, **1982**, *47*, 412–433. [DOI](https://doi.org/10.1016/0021-9991(82)90091-2)

## Implementation
The time propagation is implemented with 3 operators:

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
Note that in order to compute the correlation function it is necessary to end the propagation to the next integer time step. $N$ is thus determined by the stride for saving the results of the dynamics.

## Usage
### Preparing the potentials

The potential from *ab initio* calculations needs to be interpolated to achieve a fine grid spacing and converted to the NetCDF format. This is be done with the `interpolate.jl` script. Potential needs to be provided in the following form:
```
#   X [Bohr]       E [Hartree]
    -0.9448631     0.5890509
    -0.9259658     0.5657618
    -0.9070685     0.5429431
    -0.8881713     0.5205947
```
The user thus has to provide a simple `.txt` file with two columns containing position and energy, both in atomic units. For two-dimensional potential, the file has to contain three columns, `X Y E`.
Please note that only **regular grids** are supported, since the interpolation is done with cubic splines.

To prepare the potential file, the user has to provide an input file with the following section:
```
potfit:
    potfile: "ab-initio_scan.txt"
    NPoints: 2048   # [1024, 1024] for 2D potential
    name: "GS_potential.nc"
```
By running the command bellow the script will read the data in `ab-initio_scan.txt`, performs an interpolation with cubic splines and return a potential with 2048 grid points in file `GS_potential.nc` which is readable by `qd-engine.jl`.
```
interpolate.jl input.yml
```

---
### Running dynamics
To run the exact quantum dynamics in a split-operator formalism, some essential parameters has to be provided in the YAML input file. A mininal working example of input file is showed bellow:
```
# Input file for Quantum Dynamics Engine
# Number of dimensions:
dimensions : 1

# Basic dynamics parameters:
params:
  dt: 1.0 
  Nsteps: 80000
  stride: 10

# File with a potential (expected in NetCDF format)
potential : "GS_potential.nc"

# Mass of the particle (in amu)
mass : 14.006

# Initial condition
initWF:
  initpos: -0.13    # Position of the Gaussian WP (x0)
  freq: 1000        # frequency in cm^-1 
  #fromfile: "initWF.nc"
```
The dynamics can be then simply run by:
```
qd-engine.jl input.yml
```
#### Basic parameters
The essential parameters for the dynamics are the time step `dt` (provided in atomic units) and the number of steps `Nsteps`.
The keyword stride determines how often the dynamics are completed to the full integration step (see Implementation section above) and a point to the correlation function is added.
The amplitude and phase of the wave packet are saved to the file `WF.nc` with a given stride. Note that this file might get large **(tens of GBs)** for two-dimensional dynamics.

#### Potential energy
The potential energy operator $\hat{\mathcal{V}}(x,y)$ has to be provided in the NetCDF format created by `interpolate.jl` script (see above). The file with the potential is supplied through the keyword `potential` in the input file.

#### Kinetic energy
The form of the kinetic energy operator $\hat{\mathcal{T}}$ is restricted by the equation in the Hamiltonian section above. Only the mass of the fictitious particle needs to be specified through the keyword `mass`. In the case of the two-dimensional dynamics, the masses are provided as a list: $[\mu_x, \mu_y]$.

#### Initial condition
The script supports defining the initial condition as a real Gaussian wave packet, to achieve that the center and the width of the wave packet needs to be provided through `initpos` and `freq` keywords in the `initWF` block of the input file. The `initpos` expects value in Bohrs while `freq` a value in cm<sup>-1</sup>. For the two-dimensional dynamics, both variables need to be provided as lists. The Gaussian wave packet is then constructed according to:
```math
\begin{equation}
\psi_0(x) = \exp{ \left( -\nu\,\pi\,\mu \cdot (x-x_0)^2 \right) }
\end{equation}
```

It is furthermore possible to provide the initial wave function as a NetCDF file if the keyword `fromfile` is present. This file needs to contain dimension `x` (optionally also `y`) and variables `wfRe` (real part) and `wfIm` (imaginary part) spanned along this dimension.

Lastly, the initial conditions can be determined by an imaginary time propagation described in the following section.

---
### Imaginary time propagation
For finding the lowest eigenstate the imaginary time propagation is implemented, the wave packet is propagated by 
$\psi(t+\Delta \tau) = e^{-\frac{1}{\hbar} \hat{\mathcal{H}}\Delta \tau}\psi(t)$ where $\tau = it$.
To request the imaginary time propagation the following section has to be added to the input file, specifying only the potential on which the propagation will be carried out.
```
imagT:
  gs_potential: "GS_potential.nc"
```
After the relaxation, the file `initWF.nc` will be created, it contains the relaxed wave function and the zero-point energy.

The propagation is terminated after a convergence criterion is reached. The implemented criterion is that the change in the energy between 10 successive timesteps is smaller than $10^{-10}$ Hartree.
```math
\begin{equation}
| \langle\psi(\tau)| \hat{\mathcal{H}} | \psi(\tau)\rangle - \langle\psi(\tau+10\Delta\tau)| \hat{\mathcal{H}} | \psi(\tau+10\Delta\tau)\rangle | < 10^{-10} E_\mathrm{h}
\end{equation}
```

---
### Calculation of spectra
The energy spectrum is calculated after each run from the auto-correlation function according to:
```math
\begin{equation}
\sigma(\omega) = \frac{\omega}{2\pi} \int_0^{T} \langle \psi(0) | \psi(t) \rangle
    \cdot \mathrm{LS}(t)\,e^{i\omega t} \mathrm{d}t
\end{equation}
```
Here, $\mathrm{LS}(t)$ is a lineshape function. By default four different lineshape functions are used and results are saved in `spectrum.txt` file:

 1. No lineshape, equivalent to a rectangular window. 
 2. Hann window: $\mathrm{LS}(t)=1-\cos\left(\frac{2\pi t}{T}\right) \forall t \in (0,T); 0~elsewhere$ 
 3. Gaussian lineshape: $\mathrm{LS}(t)=\exp\left(-\frac{t^2}{0.6\cdot \mathrm{FWHM}}\right)$, the default value of FWHM is $250$ fs.
 4. Kubo lineshape: $\mathrm{LS}(t)=\exp\left(\frac{\Delta^2}{\gamma^2}\cdot(\gamma\cdot t-1+e^{-\gamma\cdot t}) \right)$, the default values of the parameters are: $\gamma = 250$ fs<sup>-1</sup>, $\Delta = \frac{4}{7}\cdot250$ fs<sup>-1</sup>.

For a calculation of vibronic spectra the script `spectra.jl` can be used to change adjust the spectral range, parameters of the lineshape functions or adding a frequency shift to the Fourier transform.
Following section can be then added to the input file:
```
# Compute spectrum from correlation function (read only by spectra.jl)
spectrum:
    maxWn: 15000
    minWn: 1500
    ZPE: "read"         # read ZPE from `initWF.nc` file
    linewidth: 550      # FWHM for Gaussian lineshape, 3/2/Delta == 1/gamma for Kubo lineshape
    outname: "spectrum_550fs"
```
The frequency shift is provided by `ZPE` keyword either as a number or as `read` which takes the ZPE from `initWF.nc` file genereted with imaginary time propagation. The frequency shift is then applied as:
```math
\begin{equation}
\sigma(\omega) = \frac{\omega}{2\pi} \int_0^{T} \langle \psi(0) | \psi(t) \rangle
    \cdot \mathrm{LS}(t)\,e^{i(\omega + \frac{E_0}{\hbar})t} \mathrm{d}t
\end{equation}
```
---
### Eigenstates
The spectral method for calculating eigenstates is implemented in `eigenstates.jl` script. The eigenstate is obtained as a Fourier transform of a moving wave packet at a given energy, as shown in the equation bellow. It is thus necessary to know the eigenenergies in advance and provide them as input.
 
```math
\begin{equation}
    \psi(x,y,E_i) = \frac{1}{T}\int\limits_0^T \psi(x,y,t) \cdot e^{ it\frac{E_i}{\hbar} } \mathrm{d}t
\end{equation}
```
The temporal evolution of the wavefunction $\psi(x,y,t)$ will be read from `WF.nc` file. Note that the final eigenstates will not be normalized (every state will have a different normalization).
The user has to provide a list of energies (in cm<sup>-1</sup>) at which the eigenstates will be calculated, the following section has to appear in the input file:
```
# Compute eigenstates using a spectral method (read only by eigenstates.jl)
eigstates:
    energies: [500, 1500, 2500, 3500] 
```

---
### Resonance Raman absorption cross-section
Heller's method for calculating the absorption cross-section for resonance Raman scattering is implemented. [2]
The central quantity of the method is the frequency dependent polarizability:
```math
\begin{equation}
\alpha_{i\rightarrow f}(\omega_I) = \int\limits_0^T 
\langle\psi_f|\hat{\mu}_{2\rightarrow 1} e^{-\frac{i}{\hbar}\hat{\mathcal{H}}_2t}\hat{\mu}_{1\rightarrow 2}|\psi_i(0)\rangle~
e^{i(\omega_I+\frac{E_0}{\hbar})t} \mathrm{d}t
\end{equation}
```
The interpretation of the formula is intuitive: the initial scattering state $\psi_i(0)$ is excited to the higher electronic state and propagated with the respective Hamiltonian $\hat{\mathcal{H}}\_2$ for time *t*. The wavepacket is then deexcited back to the ground state and projected onto the final scattering state $\psi_f$. The method is implemented with Condon approximation, *i.e.* $\hat{\mu}\neq\hat{\mu}(R)$. 
The Raman absorption cross-section is then: $\sigma(\omega_I)\propto\omega_I|\alpha_{i\rightarrow f}(\omega_I)|^2$.

The analysis is done by `spectra.jl` script, and the parameters are controlled by the following section in the input file:
```
# Resonance Raman absorption cross section (read only by spectra.jl)
Raman:
    maxWn: 20000
    minWn: 3000
    ZPE: "read"
    linewidth: 250
    finalstate: 2 # index of the final eigenstate in `eigenstates.nc`
```
The structure of the input block is similar to the `spectrum` block described above. The important part is the index of the final scattering state in the `eigenstates.nc` file, which is described above. It is also possible to provide a list of indices. Furthermore, file `WF.nc` from an excited state propagation has to be present to compute the cross-correlation function.

&nbsp;&nbsp;&nbsp;&nbsp;[2] Lee & Heller, *J. Chem. Phys.*, **1979**, *71*(12), 4777–4788. [DOI](https://doi.org/10.1063/1.438316)

---
