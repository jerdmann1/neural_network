import numpy as np

for j in range(5):
    for i in range(100):
        if np.random.uniform(0,1) < 0.5:
            print("%f %f %d" % (i, ((4 - 3*i) / 2), 1))
        else:
            if np.random.uniform(0,1) < 0.5:
                print("%f %f %d" % (i, ((4 - 3*i) / 2) + np.random.randint(1,5) , 0))
            else:
                print("%f %f %d" % (i, ((4 - 3*i) / 2) + np.random.randint(-5,-1) , 0))