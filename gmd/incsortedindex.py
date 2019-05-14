import numpy as np

class IncSortedIndex():
    def __init__(self, start_data):
        my_sorted = np.concatenate([start_data, np.array([range(0, len(start_data))]).T], axis=1)
        res = np.zeros((len(start_data),start_data.shape[1],2)) # Zeilen x Spalten x Tupel
        # Sort first window
        for col in range(start_data.shape[1]):
            res[:,col] = my_sorted[my_sorted[:, col].argsort(kind='mergesort')][:,[col,-1]]
        self.res = res
        self.window_size = len(res)
        self.col_count = start_data.shape[1]
        
    def del_and_ins_sorted(self, new_value):
        for col in range(self.col_count):
            delete_idx = np.argmin(self.res[:,col,1]) # search in indexes
            insert_idx = np.searchsorted(self.res[:,col,0], new_value[col]) # search in values

            if delete_idx==insert_idx:
                self.res[insert_idx,col] = [new_value[col], self.window_size]

            elif delete_idx < insert_idx:
                for i in range(delete_idx,insert_idx-1):
                    self.res[i,col] = self.res[i+1,col]
                self.res[insert_idx-1,col] = [new_value[col], self.window_size]

            elif delete_idx > insert_idx:
                for i in reversed(range(insert_idx,delete_idx)):
                    self.res[i+1,col] = self.res[i,col]
                self.res[insert_idx,col] = [new_value[col], self.window_size]
        self.res[:,:,1] -= 1

    @property
    def sorted(self):
        return self.res[:,:,1].astype(int)