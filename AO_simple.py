#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

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
    F_shifted[0, filter] = 0

    # Inverse Fourier transform
    F_ishifted = torch.fft.ifftshift(F_shifted)
    corrected_image = torch.fft.ifft2(F_ishifted)
    corrected_image = torch.abs(corrected_image[0])

    return corrected_image

#%%
# Load your image file here
phase_screen = load_image_as_phase_screen('./test_im.jpg', 128)

# plt.imshow(phase_screen[0])

#%
# Apply AO correction
cutoff_radius = 30 # Radius of AO correction in pixels
corrected_image = apply_ao_correction(phase_screen, cutoff_radius=cutoff_radius)

#%%
# Display the result
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(phase_screen[0], cmap='gray'), plt.title('Original Phase Screen')
plt.subplot(122), plt.imshow(corrected_image.numpy(), cmap='gray'), plt.title('After AO Correction')
plt.show()
