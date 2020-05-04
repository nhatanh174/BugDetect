from gensim.models import Word2Vec
import multiprocessing
def start(wikicorpus):
    cores=multiprocessing.cpu_count()
    # w2v_model = Word2Vec(tokenized_corpus,size=300, window=2, min_count=1, workers=10, iter=10)
    w2v_model = Word2Vec(min_count=1,
                         window=2,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores - 1,
                         iter=10)
    w2v_model.build_vocab(wikicorpus, progress_per=10000)
    w2v_model.train(wikicorpus, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1,
                    compute_loss=True)
    w2v_model.save('new_w2v_model.model')
    del w2v_model