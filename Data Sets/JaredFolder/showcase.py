from matplotlib import pyplot as plt
import numpy as np

startDate = 0
endDate = 338
data = []
for i in range(10, 170, 10):
    data.append(np.log(np.loadtxt("./STOCKDATA/BOV/TrainVal/-TRAINING-TrainSize-{0}-Start-{1}-End-{2}".format(i, startDate, endDate - i + 10))))
count = 0
for i in range(10, 170, 10):
    plt.plot(data[count], label="train-{0}".format(i))
    count += 1
plt.legend()
plt.show()


