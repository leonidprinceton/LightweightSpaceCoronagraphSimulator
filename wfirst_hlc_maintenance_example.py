##
# A dark hole maintenance example for the WFIRST Hybrid Lyot Coronagraph (based on FALCO, https://github.com/ajeldorado/falco-matlab)
# The simulation begins with a "perfect" dark hole created by FACLO. It then slowly deteriorates due to pupil-plane wavefront and DM voltages drift.
# It is assumed the Low-Order Wacefront Sensing (LOWFS) negates the contribution of Zernikes up to Z11 to the wavefront error (WFE).
# The residual low-order WFE are simulated as jitter - small fast varying changes with a non-zero average contribution to the incoherent intensity.
##

import wfirst_hlc
import numpy as np
import matplotlib.pyplot as plt
import utils

T = 200 #number of frames
broadband = True #in the absence of an IFS, the intensities across all wavelenghts (channels) are summed
EFC_regularization = 1e-7 #used when computinh the Electric Field Conjugation (EFC) gain
intensity_to_photons = 1e9 #average number of photons per contrast unit per pixel per exposure
dark_current = 0.25  #average number of dark current photons per pixel per exposure

drift_zernike_std = 1e-10 #in meters
jitter_zernike_std = 1e-9 #in meters
drift_DM_std = 1e-3 #in Volts
dither_DM_std = 1e-2 #in Volts

"""
Pupil plane wavefront drift model (to be updated when OS9 results are released).
Zenike coefficients perform a random walk with step size proportional to drift_zernike_std with scaling depended on coefficient order:
- some low order Zenikes are mitigated by LOWFS (maksed by wfirst_hlc.LOWFS_zernikes_mask); they are assumed to have zero drift post LOWFS
- the rest are scaled by the order of the polynomial squared in order to keep the "drift enegy" bounded

The LOWFS Zernike residuals contribute to low order jitter which manifests as an incoherently added intensity.
"""
drift_zernike_scaling = 1.0/(wfirst_hlc.zernikes_order+1)**2
drift_zernike_increments = np.random.normal(0, drift_zernike_std, (T, wfirst_hlc.n_zernikes))*drift_zernike_scaling*(1-wfirst_hlc.LOWFS_zernikes_mask)
drift_zernikes = np.cumsum(drift_zernike_increments, axis=0) #random walk of drift coefficients
jitter_zernike_covarinaces = ((jitter_zernike_std*drift_zernike_scaling*wfirst_hlc.LOWFS_zernikes_mask)[:wfirst_hlc.n_jitter_zernikes])**2

drift_DM_increments = np.random.normal(0, drift_DM_std, (T, wfirst_hlc.n_actuators))
drift_DM_voltages = np.cumsum(drift_DM_increments, axis=0) #random walk of DM voltages

u0 = np.zeros(wfirst_hlc.n_actuators) #zero DM command for the open-loop simulations

# store the electic fields for empirically estimating their drift covarinace (in the image plane)
E_fields = np.empty((wfirst_hlc.n_pixels,wfirst_hlc.n_channels,T), dtype=np.complex64)

# plotting functionality
plt.figure(figsize=(10,4))
plt.ion()
plt.show()
open_loop_contrasts = []
closed_loop_contrasts = []

def plot_progress(y, text):
    plt.clf()
    plt.gcf().suptitle(text, fontsize=14)
    plt.subplot(121)
    if len(open_loop_contrasts):
        plt.plot(open_loop_contrasts, "-r", label="open loop")
    if len(closed_loop_contrasts):
        plt.plot(closed_loop_contrasts, "--b", label="closed loop")
    plt.title("contrast")
    plt.xlabel("frame")
    plt.ylim((0,None))
    plt.legend(loc=4)
    plt.subplot(122)
    image = wfirst_hlc.DH_to_image(y)
    if not broadband:
        image = utils.broadband_image_to_blocks(image)
    plt.imshow(image)
    plt.title("image [photons]")
    plt.colorbar()
    plt.draw()
    plt.pause(0.01)

for t in range(T):
    # get open-loop intensities and electric fields, store them and plot progress
    I,E = wfirst_hlc.get_intensity(broadband, u0, drift_zernikes[t], drift_DM_voltages[t], jitter_zernike_covarinaces, return_E_field=True)
    E_fields[:,:,t] = E
    open_loop_contrasts.append(np.mean(I))
    y = wfirst_hlc.measurement_from_intensity(I, intensity_to_photons, dark_current)
    plot_progress(y, "step 1/3: open loop simulation (frame %d/%d)"%(t+1,T))

plot_progress(y, "step 2/3: preparing for closed loop simualtion (takes time)")

# transforms simulated complex multi-channel E-fields into real vectors be used for control (EFC)
def E_field_to_state_vector(E):
    return np.stack([E.real,E.imag],axis=2).reshape((-1,)+E.shape[2:])

# transform 3D complex Jacobian to 2D real matrix
G = E_field_to_state_vector(wfirst_hlc.jacobian)

# compute EFC gain
K_EFC = np.linalg.inv(G.T.dot(G) + EFC_regularization*np.eye(wfirst_hlc.n_actuators)).dot(G.T)

"""
Extended Kalman Filter (EKF) for estimating the electric field for a closed-loop simulation.
Running and EKF with the whole (n_pixels*n_channels*2 dimensional) vector state is infeasible.
The state is estimated individually for each pixel giving n_pixels EKFs running in paralel.
The standard EKF matrices for all pixels are concatenated along the first axis giving 3D arrays.
"""
E_states = E_field_to_state_vector(E_fields).reshape((wfirst_hlc.n_pixels, 2*wfirst_hlc.n_channels, T)) #E-fields is state-space representation for estimating drift covariance

# computing empirical covariance for the E-field drift
E_state_increments = E_states[:,:,1:] - E_states[:,:,:-1]
E_state_increments -= np.mean(E_state_increments, axis=2).reshape((wfirst_hlc.n_pixels, 2*wfirst_hlc.n_channels, 1))
Q_empirical = np.matmul(E_state_increments, E_state_increments.transpose(0,2,1))

# the covarinace used by the EKF could be Q = Q_empirical, but it works better when Q is assumed "larger" (probably because the estimation problem is non-linear)
Q = np.array([np.eye(2*wfirst_hlc.n_channels)*np.linalg.eigvalsh(q)[-1] for q in Q_empirical])

# B is the linear operation of summing over the (squares of the) electric fields to get the intensity
if broadband:
    B = np.ones((wfirst_hlc.n_pixels, 1, 2*wfirst_hlc.n_channels), dtype=np.float64)
else:
    B = np.array([np.kron(np.eye(wfirst_hlc.n_channels), [1,1])]*wfirst_hlc.n_pixels, dtype=np.float64)

# H and R hold the corresponding EKF matrices for each filter/pixel; their non-zero values are updated at each time step
H = np.zeros(B.shape, dtype=np.float64)
H_nonzero_indices = np.where(B)
B_dot_BT = np.matmul(B, B.transpose((0,2,1)))
R = np.zeros(B_dot_BT.shape, dtype=np.float64)
R_nonzero_indices = np.where(B_dot_BT)

# x_hat and P are "open-loop" state and error covariance estimates for all filters
x_hat = E_field_to_state_vector(wfirst_hlc.perfect_E).reshape((wfirst_hlc.n_pixels,-1,1))
P = Q*0

for t in range(T):
    u = -K_EFC.dot(x_hat.ravel()) #closed-loop control
    u_perfect = -K_EFC.dot(E_states[:,:,t].ravel()) #closed-loop control if open-loop E-field was perfectly known (for diagnostics)
    u_error = u - u_perfect
    u += np.random.normal(0, dither_DM_std, wfirst_hlc.n_actuators) #added dither

    # get open-loop intensities and electric fields, for diagnostics
    I,E = wfirst_hlc.get_intensity(broadband, u, drift_zernikes[t], drift_DM_voltages[t], jitter_zernike_covarinaces, return_E_field=True)
    closed_loop_contrasts.append(np.mean(I))
    # get photon counts for estimation
    y = wfirst_hlc.measurement_from_intensity(I, intensity_to_photons, dark_current)

    x_CL_hat = x_hat + G.dot(u).reshape(x_hat.shape) #estimate of the closed-loop states (E-fields for each pixels in the presence of control)
    x_CL = E_field_to_state_vector(E).reshape(x_CL_hat.shape) #actual closed-loop states for comparison
    diagnostic_str = "frame %03d, constrast %.2e, relative estimation error %.2e, relative control error %.2e"%(t, np.mean(I), np.linalg.norm(x_CL_hat-x_CL)/np.linalg.norm(x_CL), np.linalg.norm(u_error)/np.linalg.norm(u_perfect))
    print(diagnostic_str)

    # standard EKF equations broadcasted along the first axis (https://arxiv.org/abs/1902.01880)
    y_hat = np.matmul(B, x_CL_hat**2*intensity_to_photons) + dark_current
    H[H_nonzero_indices] = (2*x_CL_hat*intensity_to_photons).ravel()
    R[R_nonzero_indices] = y_hat.ravel()

    H_T = H.transpose(0,2,1)

    P = P + Q
    P_H_T = np.matmul(P, H_T)
    S = np.matmul(H, P_H_T) + R
    S_inv = np.linalg.inv(S)
    K_EKF = np.matmul(P_H_T, S_inv)
    P = P - np.matmul(P_H_T, K_EKF.transpose(0,2,1))

    x_hat = x_hat + np.matmul(K_EKF, y.reshape(y_hat.shape) - y_hat)

    plot_progress(y, "step 3/3: closed loop simulation (frame %d/%d)"%(t+1,T))

plt.ioff()
plt.show()
