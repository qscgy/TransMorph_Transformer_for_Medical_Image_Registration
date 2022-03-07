import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib

# Mostly following https://lukas-snoek.com/NI-edu/fMRI-introduction/week_1/python_for_mri.html
# A good reference for understanding how the data is stored

mri_file1 = '/playpen/aomic-id1000/derivatives/derivatives/fmriprep/sub-0100/anat/sub-0100_desc-preproc_T1w.nii.gz'
mri_file2 = '/playpen/aomic-id1000/derivatives/derivatives/fmriprep/sub-0100/anat/sub-0100_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'

img = nib.load(mri_file1)
img2 = nib.load(mri_file2)
# print(type(img))
print(img.shape)
print(img2.shape)

hdr = img.header
# print(hdr.get_zooms()) # spatial dimensions of voxels
# print(hdr.get_xyzt_units())

data = img.get_fdata()  # get volume data as numpy array
data2 = img2.get_fdata()
print(data.min(), data.max())
print(data2.min(), data2.max())
# print(data.shape)

mid_vox = data[118:121, 118:121, 108:111]
# print(mid_vox)

plt.imshow(data[120,:,:].T, cmap='gray', origin='lower')
plt.colorbar()

plt.figure()
plt.imshow(data2[193//2,:,:].T, cmap='gray', origin='lower')
plt.colorbar()
plt.show()