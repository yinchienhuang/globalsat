import os

for i in range(24):

    os.system("python scripts/preprocess.py")
    os.system("python vis/vis.py")

    print("deleting")

    os.system("del data\satellite_position1.csv")
    os.system("del data\satellite_position2.csv")

    print("renaming")
    filename = "data\satellite_position"+ str(2*i+3)+".csv"
    os.system("rename " + filename + " satellite_position1.csv")
    filename = "data\satellite_position"+ str(2*i+4)+".csv"
    os.system("rename " + filename + " satellite_position2.csv")

os.system("python scripts/preprocess.py")
os.system("python vis/vis.py")