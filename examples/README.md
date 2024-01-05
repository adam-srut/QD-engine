# Example usage
This tutorial will guide to user through the usage of the `QD-engine`. It consists of several steps:
 1. Interpolation of the potentials of the ground and excited states.
 2. Calculation of energy levels and eigenstates on the ground state surface.
 3. Calculation of vibronic spectrum by propagation of a relaxed wave packet on the excited state.
 4. Adding a frequency shift to the calculated spectrum.

### Interpolation of the potentials
Change the directory to `potfit/`. Use the `setup_potential.jl` script to create `.txt` files with ground and excited state potentials. Then:
```
interpolate.jl input_Ex.yml
interpolate.jl input_GS.yml
```
will interpolate the potentials and save them into NetCDF files. Potentials are also plotted and saved as `.png` files so you can check whether interpolation ran into problems.

### Energy levels and eigenstates in the ground state
Change the directory to  `energy-levels_eigenstates` and change and move `GS_pot.nc` file in there.
Check the settings of the dynamics in `input_1D.yml` and run the dynamics with:
```
qd-engine.jl input_1D.yml
```
When the dynamics are finished the following files are generated: 
 - `WF.nc` contains the temporal evolution of the wave packet.
 - `CF.txt` contains the autocorrelation function and `CF.gp` for plotting.
 - `spectrum.txt` contains the energy spectrum of the wave packet and `spectrum.gp` for plotting.

Inspect the spectrum, the input file should be then expanded for the identified energy levels. This is already done in `input_1D.yml`. To calculate the eigenstates with the spectral method run:
```
eigenstates.jl input_1D.yml
```
this will generate `.pdf` files with the plotted eigenstates at given energies and file `eigenstates.nc`.

### Vibronic spectrum
Move the interpolated potentials `GS_pot.nc` and `Ex_pot.nc` to `vibronic_spectrum` directory and change there.
Check the settings of the dynamics in `input_1D.yml` and run the dynamics with:
```
qd-engine.jl input_1D.yml
```
In the next step check the setting in `spectrum` block of the input file. Ensure that `ZPE: "read"` is present to take the energy of the relaxed wave packet as a frequency shift. Adjust the output name of the spectrum and width of the lineshape function. Calculate the spectrum with:
```
spectra.jl input_1D.yml
```