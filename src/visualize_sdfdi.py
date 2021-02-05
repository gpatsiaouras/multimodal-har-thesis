from datasets import UtdMhadDataset
from visualizers import frames_player

datasetSDFDI = UtdMhadDataset(modality='sdfdi')
datasetRGB = UtdMhadDataset(modality='rgb')

# Retrieve one sample, the sample is already converted to an sdfdi image so just display it
(sampleSDFDI, _) = datasetSDFDI[4]
(sampleRGB, _) = datasetRGB[4]

# Play the video first to see the action being performed. And afterwards show the SDFDI equivalent
sampleSDFDI.show()
frames_player(sampleRGB)
