import numpy as np


def read_matrix(file_name):
    with open(file_name, 'r') as file:
        for line in file:
            if line.strip() == '':
                continue
            if line[0] == '#':
                if line[2:6] == "rows":
                    _, _, size = line.split()
                    size = int(size)
                    matrix = np.zeros((size, size))
            else:
                row, col, val = line.split(' ')
                matrix[int(row)-1, int(col)-1] = val

    return matrix


def convert_to_csr(matrix):
    m, n = matrix.shape
    ICL = []
    VAL = []
    ROWPTR = []
    counter = 0

    for i in range(n): # rows
        ROWPTR.append(counter)
        for j in range(m): # columns
            val_ij = matrix[i, j]
            if abs(val_ij) < 1e-8:
                continue
            ICL.append(j)
            VAL.append(val_ij)
            counter += 1

    ROWPTR.append(counter)

    return ICL, VAL, ROWPTR
    

def sparse_cholesky(matrix):
    '''
        returns L.T matrix in CSR format
        that (L.T.)T @ L.T == matrix
    '''

    ICL = matrix[0][:]
    VAL = matrix[1][:]
    ROWPTR = matrix[2][:]
    
    n = len(ROWPTR) - 1
    
    def get_col_in_row(row, col):
        '''binary search for an index of value col in array row
            if col not in row returns index of the first bigger value than col
            if every value in row is smaller than col then returns None'''
        start = 0
        end = len(row)-1

        while start < end:
            middle = (start+end)//2
            if row[middle] < col:
                start = middle+1
            else:
                end = middle

        if row[start] == col:
            return start
        else:
            if start + 1 < len(row):
                return start + 1
            return None

    for k in range(n):
        row_start = ROWPTR[k]
        row_end = ROWPTR[k+1]

        if ICL[row_start] != k or VAL[row_start] < 0:
            raise Exception('nonpositive value on diagonal')

        VAL[row_start] **= 0.5
        dkk = VAL[row_start]

        # last row -> nothing to eliminate
        if k == n-1:
            break

        for j in range(row_start+ 1, row_end):
            VAL[j] /= dkk

        # new arrays for ICL, VAL, ROWPTR
        # starting with part of the matrix that won't be eliminated
        # later we're adding all other values after each elimination step
        new_icl = ICL[:row_end]
        new_val = VAL[:row_end]
        new_rowptr = ROWPTR[:k+2]

        vk_index = row_start + 1

        for j in range(k+1, n):
            # top_row = kth_row (not always 0th row!)
            # j_row = jth_row
            # we aim to calculate:  j_row = j_row - top_row*vk
            j_row_start = ROWPTR[j]
            j_row_end = ROWPTR[j+1]

            # we find indices in top_row ICL and j_row ICL on which value j is,
            # so we can start eliminating from there
            j_index_j_row = get_col_in_row(ICL[j_row_start:j_row_end], j)
            j_index_top_row = get_col_in_row(ICL[row_start:row_end], j)

            # if vk is 0, we just copy j_row from jth index and continue to the next row
            if vk_index >= row_end or ICL[vk_index] != j:
                if j_index_j_row is not None:
                    new_icl += ICL[j_row_start+j_index_j_row:j_row_end]
                    new_val += VAL[j_row_start+j_index_j_row:j_row_end]

                new_rowptr.append(len(new_icl))
                continue

            vk = VAL[vk_index]

            # if both top row and jth row are empty after jth index, we move onto the next row
            if j_index_j_row is None and j_index_top_row is None:
                new_rowptr.append(len(new_icl))
                continue

            # if jth row is empty after jth index we copy -vk*top_row
            if j_index_j_row is None:
                new_icl += ICL[row_start + j_index_top_row:row_end]
                new_val += [-vk*x for x in VAL[row_start +
                                               j_index_top_row:row_end]]
                new_rowptr.append(len(new_icl))
                continue
            else:
                 j_row_index = j_row_start + j_index_j_row

            # if top row is empty after jth index we just copy jth row as it is
            if j_index_top_row is None:
                new_icl += ICL[j_row_index:j_row_end]
                new_val += VAL[j_row_index:j_row_end]
                new_rowptr.append(len(new_icl))
                continue
            else:
                top_row_index = row_start + j_index_top_row


            # we iterate through top_row and j_row at the same time
            # doing the elimination
            # new non-zero values may occur
            while j_row_index < j_row_end and top_row_index < row_end:
                top_col = ICL[top_row_index]
                j_col = ICL[j_row_index]

                # nonzero value in kth row, zero in jth
                # new nonzero value
                if top_col < j_col:
                    val = -vk*VAL[top_row_index]
                    if abs(val) > 1e-8:
                        new_icl.append(top_col)
                        new_val.append(val)
                    top_row_index += 1

                # both values nonzero
                elif top_col == j_col:
                    val = VAL[j_row_index]-vk*VAL[top_row_index]
                    if abs(val) > 1e-8:
                        new_icl.append(top_col)
                        new_val.append(val)

                    top_row_index += 1
                    j_row_index += 1

                # nonzero in jth row, but zero in k
                elif top_col > j_col:
                    new_icl.append(j_col)
                    new_val.append(VAL[j_row_index])
                    j_row_index += 1

            # there might still be nonzero values in jth row
            # and just zeros in kth
            while j_row_index < j_row_end:
                new_icl.append(ICL[j_row_index])
                new_val.append(VAL[j_row_index])
                j_row_index += 1

            # there might still be nonzero values in kth row
            # and just zeros in jth
            while top_row_index < row_end:
                val = -vk*VAL[top_row_index]
                if abs(val) > 1e-8:
                    new_icl.append(ICL[top_row_index])
                    new_val.append(val)
                top_row_index += 1

            new_rowptr.append(len(new_icl))

            if vk_index < row_end and ICL[vk_index] == j:
                vk_index += 1

        ICL = new_icl
        ROWPTR = new_rowptr
        VAL = new_val

    return ICL, VAL, ROWPTR
    
    
def get_matrix_from_csr(A):
    ICL, VAL, ROWPTR = A
    VAL = VAL.copy()
    
    n = len(ROWPTR) - 1
    matrix = np.zeros((n, n))

    for row in range(n):
        for j in range(ROWPTR[row], ROWPTR[row+1]):
            matrix[row, ICL[j]] = VAL[j]
            
    return matrix
    
    
def get_ICL_VAL_from_dict(C_ROW):
    C_ROW_sorted = sorted(C_ROW.items(), key=lambda item : item[0]) # sorting by key (column index)
    ICL_C = [x[0] for x in C_ROW_sorted] # column indexes list
    VAL_C = [x[1] for x in C_ROW_sorted] # values list
    return ICL_C, VAL_C
 
    
def matmul_CSR(A, B):
    ICL_A, VAL_A, ROWPTR_A = A
    ICL_B, VAL_B, ROWPTR_B = B    
    ICL_C, VAL_C, ROWPTR_C = [], [], []

    elements_counter = 0
    for i in range(len(ROWPTR_A) - 1): # rows in A
        row_a_start_idx = ROWPTR_A[i]
        row_a_end_idx = ROWPTR_A[i+1]
        C_ROW = dict() # dict to store results of row_A * rows_B
        for j in range(row_a_start_idx,row_a_end_idx): # indexes of values in row A
            col_a = ICL_A[j]
            val_a = VAL_A[j]
            row_b_start_idx = ROWPTR_B[col_a] # column idx in A maps to row idx in B
            row_b_end_idx = ROWPTR_B[col_a + 1] # col_a + 1 to get range of value indexes in B row
            for p in range(row_b_start_idx,row_b_end_idx): # indexes of values in row B
                col_b = ICL_B[p]
                val_b = VAL_B[p]
                C_ROW[col_b] = C_ROW.get(col_b, 0) + (val_a * val_b) # update dict with val_a * val_b
                
        ICL, VAL = get_ICL_VAL_from_dict(C_ROW)
        ICL_C += ICL
        VAL_C += VAL
        ROWPTR_C.append(elements_counter)
        elements_counter += len(VAL)
    
    ROWPTR_C.append(elements_counter)
    return ICL_C, VAL_C, ROWPTR_C
    
    
