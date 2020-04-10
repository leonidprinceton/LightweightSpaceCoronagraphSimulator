import numpy as np

def broadband_image_to_blocks(image):
    n_channels = image.shape[2]
    bs = int(np.ceil(np.sqrt(n_channels)))
    blocks = [[image[:,:,i*bs+j] if i*bs+j<n_channels else image[:,:,0]*np.nan for j in range(bs)] for i in range(bs)]
    return np.block(blocks)
