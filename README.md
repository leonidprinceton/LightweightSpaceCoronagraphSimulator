# LightweightSpaceCoronagraphSimulator
Simulating realistic high-contrast space imaging instruments in a linear regime.
For now, only the Hybrid Lyot Coronagraph (HLC) of the Roman Space Telescope is supported based on [FALCO](https://github.com/ajeldorado/falco-matlab).

### Roman HLC Funcionality (wfirst_hlc/wfirst_hlc.py)
* Electric fields in 6 wavelenghts
* E-field sensitivity matrix to Deformable Mirror (DM) actuations (Jacobian)
* E-field sensitivity matrix to first 136 Zernike modes
* Dark hole mask
* Simulation of images including shot noise and dark current
* Efficient simulation of LOWFS residuals

### Dependencies
* numpy
* matplotlib (for the code example)

### Running a code example
```
python wfirst_hlc_maintenance_example.py
```
This will first generate an "open-loop" animation of deteriorating contrast due to increasing wavefront errors. Then the loop will be "closed" using an Extended Kalman Filter (EKF) and Electric Field Conjugation (EFC).
