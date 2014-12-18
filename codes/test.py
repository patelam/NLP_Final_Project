from os import system

decoder_settings = 'python beam7.py -k 10 -s 100'
reranker_settings = 'python uneven.py > weights'
rerank_settings = 'python rerank.py -w weights -k 10 -s 100 > default.output'
score_settings = 'python score_reranker.py < default.output'

system(decoder_settings + ' > train.nbest')
for i in xrange(2):
    system(reranker_settings)
    system(rerank_settings)
    system(score_settings)
    system(decoder_settings + ' -w weights > train.nbest')