import numpy as np

def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
	if len(a[0]) != len(b):
		return -1
	rows = len(a)
	res = [0] * rows
	for i in range(rows):
		res[i] = np.dot(a[i], b)
	return res