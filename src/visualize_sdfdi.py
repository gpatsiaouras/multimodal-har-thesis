from datasets import UtdMhadDataset
from tools import generate_sdfdi

from tools.optical_flow import print_sdfdi_image

dataset = UtdMhadDataset(modality='rgb', train=False)

# Retrieve one sample
(sample, _) = dataset[4]  # Subject 4, Action 1, Repetition 1
sdfdi = generate_sdfdi(sample)
print_sdfdi_image(sdfdi, continuous=False)
