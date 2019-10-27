import nmslib

class NMSlib_knn():
    """docstring for Nmslib_knn."""

    def __init__(self, featList):
        super(NMSlib_knn, self).__init__()
        self.featList = featList

        self.index = nmslib.init(method='hnsw', space='l2')
        self.index.addDataPointBatch(featList)
        self.index.createIndex({'post': 2})

    def knnSearch(self, queryFeat, k=5):

        ids, distances = self.index.knnQuery(queryFeat, k=k)

        return ids, distances
