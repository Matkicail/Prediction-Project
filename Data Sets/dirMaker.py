import os


names = ["BIS", "BOV", "EUR", "JSE", "NAS", "SP5"]
for i in names:
    os.makedirs("./WEIRDK/{0}/Testing".format(i))
    os.makedirs("./WEIRDK/{0}/TrainVal".format(i))

