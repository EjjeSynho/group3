#%%
%reload_ext autoreload
%autoreload 2

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cupy as cp

from astropy.io import fits

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

from Zernike import mask_circle, Zernike

#%%
def load_image_as_phase_screen(image_path, sampling=64):
    image = Image.open(image_path).resize((sampling,)*2).convert('L')
    transform = transforms.ToTensor()
    return transform(image)


def apply_ao_correction(phase_screen, cutoff_radius):
    # Fourier transform
    F = torch.fft.fft2(phase_screen)
    F_shifted = torch.fft.fftshift(F)

    # Create a circular high-pass filter
    rows, cols = phase_screen.shape[1], phase_screen.shape[2]
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    mask = dist_from_center <= cutoff_radius
    filter = torch.tensor(mask, dtype=torch.bool)

    # Apply the filter
    F_shifted[0, filter] *= 0.1

    # Inverse Fourier transform
    F_ishifted = torch.fft.ifftshift(F_shifted)
    corrected_image = torch.fft.ifft2(F_ishifted)

    return corrected_image.abs()


N_pupil = 256

inp_phase  = load_image_as_phase_screen('./test_im.jpg', N_pupil)
inp_phase -= inp_phase.mean()
inp_phase  = inp_phase.to(device) * 512

corrected_phase = apply_ao_correction(inp_phase, cutoff_radius=5)

plt.imshow(inp_phase[0].cpu())
plt.show()
plt.imshow(corrected_phase[0].cpu())
plt.show()

pupil = torch.tensor( mask_circle(N_pupil, N_pupil//2)[None,...] ).to(device).float()

#%%
Z_basis = Zernike(500, 64, gpu=False)


#%%
f = 400 # [m]
pixel_size = 16e-6 # [m]

λ = 700e-9 # [m]
D = 8 # [m]
img_size = 64 # [pixels]
oversampling = 2 # []


pixels_λ_D = f/pixel_size * λ/D
pad = np.round((oversampling*pixels_λ_D-1)*pupil.shape[-2]/2).astype('int')
φ_size = pupil.shape[-1] + 2*pad

padder = torch.nn.ZeroPad2d(pad.item())


def binning(inp, N):
    return torch.nn.functional.avg_pool2d(inp.unsqueeze(1),N,N).squeeze(1) * N**2 if N > 1 else inp

def OPD2PSF(λ, OPD, φ_size, padder, oversampling):  
    EMF = padder( pupil * torch.exp(2j*torch.pi/λ*OPD*1e-9) )

    lin = torch.linspace(0, φ_size-1, steps=φ_size, device=device)
    xx, yy = torch.meshgrid(lin, lin, indexing='xy')
    center_aligner = torch.exp(-1j*torch.pi/φ_size*(xx+yy)*(1-img_size%2))

    PSF = torch.fft.fftshift(1./φ_size * torch.fft.fft2(EMF*center_aligner, dim=(-2,-1)), dim=(-2,-1)).abs()**2
    cropper = slice(φ_size//2-(img_size*oversampling)//2, φ_size//2+round((img_size*oversampling+1e-6)/2))

    PSF = binning(PSF[...,cropper,cropper], oversampling)
    return PSF


#%%
# Load .fits file
def load_fits_as_phase_screen(fits_path):
    # Load FITS file
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
    
    # Ensure the data is in native byte order
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().newbyteorder()

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(data).float()
    tensor = tensor.unsqueeze(0)  # Add a channel dimension
    return tensor

X = load_fits_as_phase_screen('./phase_screen.fits')
X = X * 700.0 / 2 / np.pi # [nm]

X = X.to(device)
#%%

def GetPSF(phase_cube):
    return OPD2PSF(λ, phase_cube, φ_size, padder, oversampling)
    
# PSF_0 = GetPSF(inp_phase)
PSF_0 = GetPSF(-X)
# PSF_1 = GetPSF(corrected_phase)

plt.imshow(torch.log(PSF_0[0].abs().cpu()))
plt.show()
# plt.imshow(torch.log(PSF_1[0].abs().cpu()))
# plt.show()

#%%
# Apply AO correction
cutoff_radius = 5 # Radius of AO correction in pixels
corrected_image = apply_ao_correction(X, cutoff_radius=cutoff_radius)

#%
# Display the result
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(inp_phase[0], cmap='gray'), plt.title('Original Phase Screen')
plt.subplot(122), plt.imshow(corrected_image.numpy(), cmap='gray'), plt.title('After AO Correction')
plt.show()
