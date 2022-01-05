from itertools import chain

def rowptr_from_list(val_list):
    ROWPTR = []
    counter = 0
    for row in val_list:
        ROWPTR.append(counter)
        counter += len(row)
    ROWPTR.append(counter)
    return ROWPTR

def sparse_cholesky_permutation(matrix, permutation):
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
    
    icl_list = [[] for _ in range(len(permutation))]
    val_list = [[] for _ in range(len(permutation))]
    
    for k_idx, k in enumerate(permutation):
        row_start = ROWPTR[k]
        row_end = ROWPTR[k+1]

        if ICL[row_start] != k or VAL[row_start] < 0:
            print("ICL[{}]={} != {} or VAL[{}]={} < 0".format(row_start, ICL[row_start], k, row_start, VAL[row_start]))
            raise Exception('nonpositive value on diagonal')

        VAL[row_start] **= 0.5
        dkk = VAL[row_start]
        
        for j in range(row_start+ 1, row_end):
            VAL[j] /= dkk

        icl_list[k] = ICL[row_start:row_end]
        val_list[k] = VAL[row_start:row_end]
        
        for k_non_visited in permutation[k_idx + 1:]:
            icl_list[k_non_visited] = []
            val_list[k_non_visited] = []

        
        # last row -> nothing to eliminate
        if k == permutation[-1]:
            break
            
        vk_index = row_start + 1

        for j in permutation[k_idx + 1:]:
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
                    icl_list[j] += ICL[j_row_start+j_index_j_row:j_row_end]
                    val_list[j] += VAL[j_row_start+j_index_j_row:j_row_end]
                continue

            vk = VAL[vk_index]

            # if both top row and jth row are empty after jth index, we move onto the next row
            if j_index_j_row is None and j_index_top_row is None:
                continue

            # if jth row is empty after jth index we copy -vk*top_row
            if j_index_j_row is None:
                icl_list[j] += ICL[row_start + j_index_top_row:row_end]
                val_list[j] += [-vk*x for x in VAL[row_start +
                                               j_index_top_row:row_end]]
                continue
            else:
                 j_row_index = j_row_start + j_index_j_row

            # if top row is empty after jth index we just copy jth row as it is
            if j_index_top_row is None:
                icl_list[j] += ICL[j_row_index:j_row_end]
                val_list[j] += VAL[j_row_index:j_row_end]
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
                        icl_list[j].append(top_col)
                        val_list[j].append(val)   
                    top_row_index += 1

                # both values nonzero
                elif top_col == j_col:
                    val = VAL[j_row_index]-vk*VAL[top_row_index]
                    if abs(val) > 1e-8:
                        icl_list[j].append(top_col)
                        val_list[j].append(val)   
                    top_row_index += 1
                    j_row_index += 1

                # nonzero in jth row, but zero in k
                elif top_col > j_col:
                    icl_list[j].append(j_col)
                    val_list[j].append(VAL[j_row_index])      
                    j_row_index += 1

            # there might still be nonzero values in jth row
            # and just zeros in kth
            while j_row_index < j_row_end:
                icl_list[j].append(ICL[j_row_index])
                val_list[j].append(VAL[j_row_index]) 
                j_row_index += 1

            # there might still be nonzero values in kth row
            # and just zeros in jth
            while top_row_index < row_end:
                val = -vk*VAL[top_row_index]
                if abs(val) > 1e-8:
                    icl_list[j].append(ICL[top_row_index])
                    val_list[j].append(val) 
                top_row_index += 1

            if vk_index < row_end and ICL[vk_index] == j:
                vk_index += 1
        
        
        ROWPTR = rowptr_from_list(val_list)
        ICL = list(chain.from_iterable(icl_list))
        VAL = list(chain.from_iterable(val_list))

    return ICL, VAL, ROWPTR


