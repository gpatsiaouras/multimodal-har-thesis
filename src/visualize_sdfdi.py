from datasets import MmactDataset
from visualizers import frames_player

datasetSDFDI = MmactDataset(modality='sdfdi')
datasetRGB = MmactDataset(modality='video')

# Retrieve one sample, the sample is already converted to an sdfdi image so just display it
(sampleSDFDI, _) = datasetSDFDI[4]
(sampleRGB, _) = datasetRGB[4]

# Play the video first to see the action being performed. And afterwards show the SDFDI equivalent
sampleSDFDI.show()
frames_player(sampleRGB)
