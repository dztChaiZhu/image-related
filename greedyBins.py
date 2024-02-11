
import numpy as np
import heapdict 
import time 

class greedyBins():
    """
    binning N=[startNumber] distributions of length d=[alpSize] into M=[targetNumber] classes.
    i.e., reducing N classes of distributions P(x|i), i=1,..,N to its M clusters. (2<N, M<N)  
    """
    def __init__(self, alpSize:int, startNumber:int, targetNumber):
        self._alpSize = alpSize
        assert(startNumber > 2)
        assert(targetNumber <= startNumber)
        self._initN = startNumber
        self._targetN = targetNumber
        #we use a numpy data structure to manage the searching tree. the initial nodes are _initN many freqencies with
        #no parents. each node has a unique id as the idx when its created. if two nodes idx_1, idx_2 are merged, the new node will
        #have left=idx_1, right=idx_2 as its parents. 
        #'score', 'avl', 'weight', are internal parameters, while 'freqs' is the numpy vector that represent distribution
        self.nodes_type = np.dtype({'names' : ['id', 'left', 'right', 'score', 'avl', 'weight', 'freqs'],
                                   'formats': ['<i4', '<i4', '<i4',  '<f4', '?',   '<f4',    '%s<f4'%alpSize]}, align=True)
        initSize = startNumber + (startNumber*(startNumber-1)//2) # N singletons + C(N, 2) init working space.
        self._data = np.zeros(initSize, dtype = self.nodes_type)
        
        #init the heap for finding the pair of nodes with smallest merge cost. 
        self._heap = heapdict.heapdict() #node=(id, left, right) will be push in, with values the merging cost. 
        self.curId = 0 # + 1 for whenever a new node is created
        self.totalCost = 0.0 #sum up merging cost
    
    def __str__(self):
        return "Total merging cost: %.3f"%self.totalCost
    
    def _entropy(self, p):
        p = p[p > 0] #to avoid log2(0)
        return -np.sum(p*(np.log2(p)))
    
    def _mergeCost(self, aa, bb): #this is only a symmetric version of KL divergence with weights
        wa, wb= aa['weight'], bb['weight'] 
        fa, fb= aa['freqs'], bb['freqs']
        ena, enb = aa['score'], bb['score']
        wab = wa + wb
        fab = (wa*fa + wb*fb)/wab
        entropy = self._entropy(fab)
        delta = wab*(entropy) - wa*(ena) - wb*(enb)
        return wab, fab, entropy, float(delta)
    
    def loadIn(self, weight_list, freq_array):
        #all initial singletons are available (to be pop out), with parents set to -1.
        self._data[:self._initN] = [(i, -1, -1, self._entropy(freq_array[i]), True, weight_list[i], freq_array[i]) for i in range(self._initN)]
        self.curId += self._initN #update curId 
        #compute the pairwise merging cost of each pair of nodes. 
        for j in range(self._initN): 
            for k in range(j+1, self._initN):
                left = self._data[j]
                right = self._data[k]
                pab, fab, entropy, delta = self._mergeCost(left, right)
                #assign (left, right) to curld. push into heap.
                self._heap[(self.curId, left["id"], right["id"])] = delta
                #save data, "avl" default = false since we 
                self._data[self.curId] = (self.curId, left["id"], right["id"], entropy, False, pab, fab[:])
                self.curId += 1
    
    #pop a valid node from the heap with smallest merging cost 
    def popNode(self):
        getIdx = lambda x: np.nonzero(self._data["id"] == x)
        node, delta = self._heap.popitem() #pop the node with minimal delta!
        curId, leftId, rightId = node #unpack ids of current, left, right
        idx = getIdx(curId)
        lidx = getIdx(leftId)
        ridx = getIdx(rightId)
        # it has to be a node with avaliable parents.
        while (not self._data[lidx]["avl"]) or (not self._data[ridx]["avl"]):
            node, delta = self._heap.popitem()
            curId, leftId, rightId = node
            idx = getIdx(curId)
            lidx = getIdx(leftId)
            ridx = getIdx(rightId)
           
        self.totalCost += delta
        #set this node to be avaliable, and parents to be not avaliable 
        #WARNING: assigning value can only in ["avl"]["idx"] order! 
        self._data["avl"][lidx] = False
        self._data["avl"][ridx] = False
        self._data["avl"][idx] = True
        
        #remove nodes that are not longer needed after the merge 
        #these are nodes whose left or right == the poped left or right, except for the current one
        mask1 = (self._data['left'] == leftId)
        mask2 = (self._data['left'] == rightId)
        mask3 = (self._data['right'] == leftId)
        mask4 = (self._data['right'] == rightId)
        mask = np.logical_not(mask1|mask2|mask3|mask4)
        mask[idx] = True #except the current one
        self._data = self._data[mask]
        return curId
    
    #add current node to avl data, and push merged result into the heap 
    def addNode(self, curId):
        avaList = np.nonzero(self._data["avl"] == True)[0]
        cur = self._data[np.nonzero(self._data["id"]==curId)]
        #size - 1 since we do not need add current node
        temp = np.zeros(avaList.size-1, dtype = self.nodes_type)
        ptr = 0
        for idx in avaList:
            foo = self._data[idx]
            if foo["id"] != curId:
                pab, fab, entropy, delta = self._mergeCost(cur, foo)
                self._heap[(self.curId, curId, foo["id"])] = delta
                temp[ptr] = (self.curId, curId, foo["id"], entropy, False, pab, fab[:])
                ptr += 1
                self.curId += 1
        #concatenate temp to data, this together with the removing nodes in popNode() maintain the database for this program to be not
        #too large. indeed, one can observe significant runtime improvement when _data is managed, as python will automatically free memories
        #that are no longer attached to class parameters. 
        self._data = np.concatenate((self._data, temp), dtype = self.nodes_type)
    
    def _walkBack(self, node, tree):
        cur, left, right = node #unpack
        if left == -1 and right == -1:
            return [cur]
        #if cur is not a singleton, push in left and right as child
        child = [left, right]
        root = []
        #The nice part, walk back from the current leaf to all nodes that are connects to it. 
        while len(child) > 0:
            cur, left, right = tree[np.nonzero(tree["id"]==child[0])][0]
            if left == -1 and right == -1: #cur is a root 
                root.append(cur)
            else:
                child.append(left)
                child.append(right)
            child.remove(cur)
        root.sort()
        return root
    
    def getResult(self):
        tree = self._data[["id", "left", "right"]]
        finalNode = tree[np.nonzero(self._data["avl"] == True)]
        result = []
        for node in finalNode:
            result.append(self._walkBack(node, tree)) 
        return result
    
    def getMergedDistribution(self):
        dataSet = self._data[np.nonzero(self._data["avl"] == True)]
        return dataSet["weight"], dataSet["freqs"]
    
    def RUN(self):
        ct = self._initN
        while ct> self._targetN:
            self.addNode(self.popNode())
            ct -= 1
        return self.getMergedDistribution()


"""
TESTING
"""
def sampleFromGaussian(mean, stv, sampleSize, bins_in = np.arange(257)):
    sample = np.random.normal(mean, stv, size = sampleSize)
    sample[sample < bins_in[0]] = bins_in[0]
    sample[sample > bins_in[-1]] = bins_in[-1]
    counts, _bin_edges = np.histogram(sample, bins = bins_in)
    return counts/np.sum(counts)

if __name__ == "__main__":
    # Initiate a dummy data set of Gaussian distributions with different mean values and same
    # variance that is small enough so that it will be easy to see the diffenence.
    # Thus those with similar means should be clustered together. In this case, we use a random sampling to decide
    # the ID number of those distributions.
    _INIT_NODE_NUMBER = 256 #total number of ditributions
    _ALPHABET_SIZE = 256
    _BINNED_SIZE = 8
    #we use a uniform weight in this example
    weight = np.ones(_INIT_NODE_NUMBER)*(1/_INIT_NODE_NUMBER)
    classes = np.random.randint(0, _BINNED_SIZE, size=_INIT_NODE_NUMBER)
    print("Estimated Grouping")
    for i in range(_BINNED_SIZE):
        print(np.where(classes == i)[0])
    #generate distributions.
    stv =_ALPHABET_SIZE/(_BINNED_SIZE+1)
    classToMean = stv*(np.arange(1, _BINNED_SIZE+1))
    means = [classToMean[x] for x in classes]
    
    #create input data
    freqs = np.zeros( _INIT_NODE_NUMBER*_ALPHABET_SIZE, dtype = np.float32).reshape(-1, _ALPHABET_SIZE)
    for j in range(_INIT_NODE_NUMBER):
        freqs[j,:] = sampleFromGaussian(means[j], stv, 1000, bins_in = np.arange(_ALPHABET_SIZE+1))
    
    print("Start Clustering")
    mytree = greedyBins( _ALPHABET_SIZE, _INIT_NODE_NUMBER, _BINNED_SIZE)
    mytree.loadIn(weight, freqs)
    print("Init Done")
    start = time.perf_counter_ns()
    mytree.RUN()
    groups = mytree.getResult()
    print("Result of Clustering")
    for subclass in groups:
        print("#", subclass)
    span = (time.perf_counter_ns()-start)/1e9
    print("time: %.2f s"%span)
    
