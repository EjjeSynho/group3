#%%
%reload_ext autoreload
%autoreload 2

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')


def mask_circle(N, r, center=(0,0), centered=True):
    factor = 0.5 * (1-N%2)
    if centered:
        coord_range = np.linspace(-N//2+N%2+factor, N//2-factor, N)
    else:
        coord_range = np.linspace(0, N-1, N)
    xx, yy = np.meshgrid(coord_range-center[1], coord_range-center[0])
    pupil_round = np.zeros([N, N], dtype=np.int32)
    pupil_round[np.sqrt(yy**2+xx**2) < r] = 1
    return pupil_round


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


N_pupil = 128

inp_phase  = load_image_as_phase_screen('./test_im.jpg', 128)
inp_phase -= inp_phase.mean()
inp_phase  = inp_phase.to(device) * 512

corrected_phase = apply_ao_correction(inp_phase, cutoff_radius=5)

plt.imshow(inp_phase[0].cpu())
plt.show()
plt.imshow(corrected_phase[0].cpu())
plt.show()

#%%
pupil = torch.tensor( mask_circle(128, 64)[None,...] ).to(device).float()

#%
f = 600 # [m]
pixel_size = 24e-6 # [m]

λ = 1000e-9 # [m]
D = 8.1 # [m]
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


def GetPSF(phase_cube):
    return OPD2PSF(λ, phase_cube, φ_size, padder, oversampling)
    
PSF_0 = GetPSF(inp_phase)
PSF_1 = GetPSF(corrected_phase)

plt.imshow(torch.log(PSF_0[0].abs().cpu()))
plt.show()
plt.imshow(torch.log(PSF_1[0].abs().cpu()))
plt.show()

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
