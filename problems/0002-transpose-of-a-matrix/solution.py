def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	rows, cols = len(a[0]), len(a)
	res = [[0] * cols for _ in range(rows)]
	for i in range(rows):
		for j in range(cols):
			res[i][j] = a[j][i]
	return res