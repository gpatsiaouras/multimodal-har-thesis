from datasets import UtdMhadDataset
from visualizers import frames_player

datasetSDFDI = UtdMhadDataset(modality='sdfdi', train=False)
datasetRGB = UtdMhadDataset(modality='rgb', train=False)

# Retrieve one sample, the sample is already converted to an sdfdi image so just display it
(sampleSDFDI, _) = datasetSDFDI[4]  # Subject 4, Action 1, Repetition 1
(sampleRGB, _) = datasetRGB[4]  # Subject 4, Action 1, Repetition 1

# Play the video first to see the action being performed. And afterwards show the SDFDI equivalent
frames_player(sampleRGB)
sampleSDFDI.show()
