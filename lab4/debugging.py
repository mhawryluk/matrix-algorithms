import numpy as np


def cholesky_LLT(matrix):
    A = matrix.copy()
    n = A.shape[0]

    for k in range(n):
        if abs(A[k, k]) < 1e-8:
            raise ValueError('singular matrix')

        vk = A[k, k+1:n]
        A[k, k] **= 0.5
        dkk = A[k, k]
        A[k, k+1:n] /= dkk

        for j in range(k+1, n):
            A[j, j:n] -= A[k, j:n]*vk[j-k-1]

        print(A)
    return np.triu(A).T


def sparse_cholesky(matrix):
    ICL, VAL, ROWPTR = matrix
    n = max(ICL) + 1

    for k in range(n):
        row_start = ROWPTR[k]
        row_end = ROWPTR[k+1]

        dkk_index = get_col_in_row(ICL[row_start:row_end], k)
        VAL[row_start + dkk_index] **= 0.5
        dkk = VAL[row_start + dkk_index]

        if k == n-1:
            break

        for j in range(row_start+dkk_index + 1, row_end):
            VAL[j] /= dkk

        vk_icl = ICL[row_start + dkk_index + 1:row_end]
        vk_val = VAL[row_start + dkk_index + 1:row_end]

        new_icl = ICL[:row_end]
        new_val = VAL[:row_end]
        new_rowptr = ROWPTR[:k+2]

        vk_index = 0

        for j in range(k+1, n):
            # print_CSR_matrix((new_icl, new_val, new_rowptr))
            if vk_index >= len(vk_icl) or vk_icl[vk_index] != j:
                vk = 0
            else:
                vk = vk_val[vk_index]

            j_row_start = ROWPTR[j]
            j_row_end = ROWPTR[j+1]

            j_index_j_row = get_col_in_row(ICL[j_row_start:j_row_end], j)
            j_index_top_row = get_col_in_row(ICL[row_start:row_end], j)

            # todo - naprawić:
            if j_index_j_row is not None:
                j_row_index = j_row_start + j_index_j_row
            else:
                new_icl += ICL[row_start + j_index_top_row:row_end]
                new_val += [-vk*x for x in VAL[row_start +
                                               j_index_top_row:row_end]]
                new_rowptr.append(len(new_icl))
                continue

            if j_index_top_row is not None:
                top_row_index = row_start + j_index_top_row
            else:
                new_icl += ICL[j_row_index:j_row_end]
                new_val += VAL[j_row_index:j_row_end]
                new_rowptr.append(len(new_icl))
                continue
            ##

            while j_row_index < j_row_end and top_row_index < row_end:
                top_col = ICL[top_row_index]
                j_col = ICL[j_row_index]

                if top_col < j_col:
                    val = -vk*VAL[top_row_index]
                    if abs(val) > 1e-8:
                        new_icl.append(top_col)
                        new_val.append(val)
                    top_row_index += 1

                elif top_col == j_col:
                    val = VAL[j_row_index]-vk*VAL[top_row_index]
                    if abs(val) > 1e-8:
                        new_icl.append(top_col)
                        new_val.append(val)

                    top_row_index += 1
                    j_row_index += 1

                elif top_col > j_col:
                    new_icl.append(j_col)
                    new_val.append(VAL[j_row_index])
                    j_row_index += 1

            while j_row_index < j_row_end:
                new_icl.append(ICL[j_row_index])
                new_val.append(VAL[j_row_index])
                j_row_index += 1

            while top_row_index < row_end:
                val = -vk*VAL[top_row_index]
                if abs(val) > 1e-8:
                    new_icl.append(ICL[top_row_index])
                    new_val.append(val)
                top_row_index += 1

            new_rowptr.append(len(new_icl))
            if vk_index < len(vk_icl) and vk_icl[vk_index] == j:
                vk_index += 1

        ICL = new_icl
        ROWPTR = new_rowptr
        VAL = new_val

        print_CSR_matrix((ICL, VAL, ROWPTR))
        print()

    print_CSR_matrix((ICL, VAL, ROWPTR))
    print()

    return ICL, VAL, ROWPTR


def print_CSR_matrix(A):
    print(get_matrix_from_CSR(A))


def get_matrix_from_CSR(A):
    ICL, VAL, ROWPTR = A
    VAL = VAL.copy()

    n = len(ROWPTR) - 1
    matrix = np.zeros((n, n))

    for row in range(n):
        for j in range(ROWPTR[row], ROWPTR[row+1]):
            matrix[row, ICL[j]] = VAL[j]

    return matrix


def get_col_in_row(ICL_row, col):
    # zmienić na bin_search
    for i, x in enumerate(ICL_row):
        if x >= col:
            return i


def convert_to_csr(matrix):
    m, n = matrix.shape
    ICL = []
    VAL = []
    ROWPTR = []
    counter = 0

    for i in range(n):  # rows
        ROWPTR.append(counter)
        for j in range(m):  # columns
            val_ij = matrix[i, j]
            if abs(val_ij) < 1e-8:
                continue
            ICL.append(j)
            VAL.append(val_ij)
            counter += 1

    ROWPTR.append(counter)

    return ICL, VAL, ROWPTR


# matrix = np.array([[4, 0, 1], [0, 4, 0], [1, 0, 4]], dtype=float)
matrix = np.array([
    [25, 0, 0, 5, 10],
    [0, 36, 0, 6, 0],
    [0, 0, 9, 0, 3],
    [0, 0, 0, 100, 0],
    [0, 0, 0, 0, 14]
], dtype=float)
cholesky_LLT(matrix)
print('---------')
sparse_cholesky(convert_to_csr(matrix))
