import numpy as np
from matplotlib import pyplot as plt


data = []
startDate = 2524
endDate = 3032
exchange = "JSE"
for i in range(10, 100, 10):
    temp = np.loadtxt("./" + exchange + "/TrainVal/-TRAINVAL-TrainSize-{0}-Start-{1}-End-{2}".format(i, startDate-i, endDate))
    data.append(temp)

count = 10
for i in data:
    plt.plot(i, label = "TrainSize-{0}".format(count))
    count += 10
plt.legend()
plt.title("{0} StartDate {1} - EndDate {2}".format(exchange, startDate, endDate))
plt.xlabel("# Trading Days")
plt.ylabel("Return")
plt.show()
    