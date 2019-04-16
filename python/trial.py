import numpy as np

list2 = [np.asarray([2, 4]), np.asarray([2, 4])]
list3 = [np.asarray([2, 4]), np.asarray([2, 4])]
list1 = [list2, list3]
print(np.sum(list1, axis=0))
