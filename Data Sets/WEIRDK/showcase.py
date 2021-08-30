import numpy as np
from matplotlib import pyplot as plt


data = []
startDate = 200
endDate = 708
exchange = "JSE"
sizes = [10, 110, 190]
for i in sizes:
    temp = np.loadtxt("./" + exchange + "/TrainVal/-TRAINVAL-TrainSize-{0}-Start-{1}-End-{2}".format(i, startDate-i, endDate))
    data.append(temp)

count = 0
for i in data:
    plt.plot(i, label = "TrainSize-{0}".format(sizes[count]))
    count += 1
plt.legend()
plt.title("{0} StartDate {1} - EndDate {2}".format(exchange, startDate, endDate))
plt.xlabel("# Trading Days")
plt.ylabel("Return")
plt.show()
    