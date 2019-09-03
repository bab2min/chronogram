import chronogram as cg

class DataReader:
    '''
    Since the whole corpus may be iterated 
    more than once during build_vocab or train, 
    so using DataReader is more efficient than
    loading everything into an in-memory list.
    '''
    def __init__(self, filepath):
        self.filepath = filepath
    
    def __iter__(self):
        for line in open(self.filepath):
            # sample data is consist of two fields, time and sentence.
            fields = line.split('\t', 1)
            if len(fields) < 2: continue
            words = fields.split()
            time = float(fields[1])
            
            # yielding words-time pair
            yield words, time
            

# create model with 300 dimensions and 8 ordered polynomial
mdl = cg.Chronogram(m=300, l=8)
data_reader = DataReader('sample.txt')
# build vocabulary before training. 
# words whose count is less than 10 
#  will be removed from trainin data.
# workers=0 means it uses all available cores.
mdl.build_vocab(reader=data_reader, min_cnt=10, workers=0)

# we initialize our model's weights first. 
# the method initializes all weights following word2vec skip-gram
#  with window size = 4.
# we can skip initializing step and go directly into training.
mdl.initialize(reader=data_reader, workers=0, window_len=4)

# train the model.
# we will iterate all training data with 10 epochs
mdl.train(reader=data_reader, workers=0, window_len=4, epochs=10)

# and save the final model.
mdl.save('chronogram-sample.mdl')

# we can load the model from file.
mdl = Chronogram.load('chronogram-sample.mdl')

# we can find similar words at 2000 to 'model' at 1980
print('Similar words of "model" at 1980')
for word, similarity, p in mdl.most_similar(('model', 1980), time=2000):
    print('%s %f' % (word, similarity))

# we can estimate the time of unknown text.
sample = "sample sentence written at 1990".split()
print(' '.join(sample))
est_time = mdl.estimate_time(sample, window_len=4, workers=0, min_t=1960, max_t=2010)
print('Estimated Time: %f' % est_time)

