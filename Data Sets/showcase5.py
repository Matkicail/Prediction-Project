import numpy as np
from matplotlib import pyplot as plt


mixedModel = np.loadtxt("./TrainVal-start-200-end-708-JSE-sizesmixed-model-10-110.txt")
smallComponent = np.loadtxt("./WEIRDK/BOV/TrainVal/-TRAINVAL-TrainSize-10-Start-190-End-708")
medComponent = np.loadtxt("./WEIRDK/BOV/TrainVal/-TRAINVAL-TrainSize-120-Start-80-End-708")
largeComponent = np.loadtxt("./WEIRDK/BOV/TrainVal/-TRAINVAL-TrainSize-190-Start-10-End-708")

logGraphs = False
if logGraphs:
    smallComponent = np.log(smallComponent)
    medComponent = np.log(medComponent)
    largeComponent = np.log(largeComponent)

plt.plot(smallComponent, label="small")
plt.plot(medComponent, label="medium")
plt.plot(largeComponent, label="large")
plt.plot(mixedModel, label="mixed-model")
plt.legend()
plt.show()