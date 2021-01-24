from datasets import MmactDataset, UtdMhadDataset
from visualizers import frames_player

datasetSDFDI = MmactDataset(modality='sdfdi', train=True)
# datasetRGB = MmactDataset(modality='video', train=False)

# Retrieve one sample, the sample is already converted to an sdfdi image so just display it
(sampleSDFDI, _) = datasetSDFDI[4]  # Subject 4, Action 1, Repetition 1
# (sampleRGB, _) = datasetRGB[4]  # Subject 4, Action 1, Repetition 1

# Play the video first to see the action being performed. And afterwards show the SDFDI equivalent
sampleSDFDI.show()
# frames_player(sampleRGB)
