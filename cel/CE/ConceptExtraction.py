class ConceptExtraction:
    def __init__(self, dimension):
        self.dimension = dimension

    def fit(self, dataloader):
        raise Exception ("Not Implemented")
    
    def eval(self, data):
        raise Exception ("Not Implemented")
    
    def report(self, dataloader):
        raise Exception ("Not Implemented")
    
    def save(self):
        raise Exception("Not implemented")
    
    def load(self):
        raise Exception("Not implemented")