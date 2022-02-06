import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib

# Mostly following https://lukas-snoek.com/NI-edu/fMRI-introduction/week_1/python_for_mri.html
# A good reference for understanding how the data is stored

mri_file = '/playpen/Downloads/sub-0011 anat sub-0011_run-1_T1w.nii.gz'
img = nib.load(mri_file)
print(type(img))
print(img.shape)

hdr = img.header
print(hdr.get_zooms()) # spatial dimensions of voxels
print(hdr.get_xyzt_units())

data = img.get_fdata()  # get volume data as numpy array
print(data.shape)

mid_vox = data[118:121, 118:121, 108:111]
print(mid_vox)

plt.imshow(data[80,:,:].T, cmap='gray', origin='lower')
plt.show()