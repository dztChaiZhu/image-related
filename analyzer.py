import numpy as np
import random #seed, getrandbit
from zlib import crc32 
from struct import unpack
import matplotlib.pyplot as plt

class Analyzer():
    """
    This class is used to contain basic statistics of a given data, other analysis tools can use it as parents to adopt
    computed statistics. 
    """
    _COUNTS_DTYPE = np.uint32
    _SYMBOL_DTYPE = np.uint8
    _MAX_COUNT = np.iinfo(_COUNTS_DTYPE).max
    _MAX_SYMBOL_VALUE = np.iinfo(_SYMBOL_DTYPE).max
    def __init__(self, data:np.ndarray)->None:
        #get the very basic: distribution
        ap, ct = np.unique(data, return_counts = True)
        assert np.max(ct)<Analyzer._MAX_COUNT
        assert ap[-1]<=Analyzer._MAX_SYMBOL_VALUE
        self.ap = ap.astype(Analyzer._SYMBOL_DTYPE)
        self.ct = ct.astype(Analyzer._COUNTS_DTYPE)
        self.sort_ct = np.argsort(self.ct).astype(Analyzer._COUNTS_DTYPE)
        self.maxSymValue = int(self.ap[-1])
        self.maxCtSym = int(self.ap[self.sort_ct[-1]])
        self.totalCt = int(np.sum(self.ct))
        self.crc = crc32(data.tobytes())
        return None
    
    def get_entropy(self)->float: 
        ctt = self.ct #do not need to worry 0 in ctt since np.unique()
        if ctt.size == 1:
            return 0.0
        L = self.totalCt
        return float(np.sum(ctt*(np.log2(L) - np.log2(ctt)))/L)
    
    def get_importantSymbols(self, number):
        if number < (self.sort_ct).size:
            return self.ap[self.sort_ct[-number:]]
        else:
            return self.ap
    

class wordSample(Analyzer):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        self.data = data
        random.seed(1987)
        
    
    #sample length = word_size strings, with sample distance = search_stride.    
    #then a frequency table of possible prefix is created (so that the word can be long, but 
    #the runtime can be adjusted), upon which an importance sampling is based. 
    #from the result of importance sampling, a set of words is return as the "significant set" that represent
    #the data set. 
    def prefixSample_internal(self, search_stride, word_size, prefix_len = 3, threshold = 0.7, max_sample_size = 100 ):
        #transform data into word sequence
        data = self.data
        base_step = data.strides[0]
        row_size = (data.size - word_size)//search_stride
        if (data.size - row_size*search_stride) >= word_size:
            row_size += 1
        assert row_size > 0
        assert prefix_len <= word_size
        foo = np.lib.stride_tricks.as_strided(data, shape = (row_size, word_size), strides = (search_stride*base_step, base_step))
        #get frequency of prefix
        prefix = foo[:,:prefix_len]
        ap, ct = np.unique(prefix, return_counts=True, axis=0)
        ct_sort = np.argsort(ct)
        #create an importance sample plan, sample 
        stop = 0
        count = 0
        threshold_count = threshold*np.sum(ct)
        for i in range(1, len(ct_sort)):
            count += ct[ct_sort[-i]]
            if count > threshold_count:
                stop = i
                break
        impt_ct = ct[ct_sort[-stop:]]
        impt_ap = ap[ct_sort[-stop:]]
        possible_size = np.sum(impt_ct)
        sample_size = max_sample_size if max_sample_size<possible_size else possible_size
        impt_sample_plan = np.floor(impt_ct*(sample_size/possible_size))
        impt_sample_plan[-1] += sample_size - int(np.sum(impt_sample_plan)) #put residue counts from floor to most important
        #create the word set
        word_set = set()
        for i in range(1, stop):
            mask = np.all(prefix == impt_ap[-i], axis = 1)
            word_list = foo[mask]
            start = len(word_set)
            end = impt_sample_plan[-i]
            for idx in range(1, len(word_list)+1):
                word = word_list[-idx]
                word_set.add(bytes(word))
                if len(word_set)-start >= end:
                    break
        if len(word_set)>1:
            return list(word_set)
        return None
    
    def getFingerPrint(self, wordList: list) -> int:
        #uses hash values to represent bytes
        crcs = [crc32(item) for item in wordList]
        shingles = np.array(crcs, dtype = np.uint32).reshape(-1, 1)
        
       


if __name__ == "__main__":
    dummy = np.random.randint(0,256, size = 177, dtype = np.uint8)
    axa = wordSample(dummy)
    print(dummy)
    ana = axa.prefixSample_internal(4, 7)
    axa.getFingerPrint(ana)
    





    


