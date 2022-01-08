import numpy as np

second_stack = np.load('second_stack.npy')
print(np.unique(second_stack, return_counts=True))
