#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple
import numpy
from bleu import bleu_stats, smoothed_bleu
from operator import add, sub

optparser = optparse.OptionParser()
translation_candidate = namedtuple("candidate", "sentence, scores, features, smoothed_bleu")
optparser.add_option("-n", "--nbest", dest="nbest", default="train_small.nbest", help="N-best file")
optparser.add_option("-l","--length", dest="length", default=sys.maxint, help="Number of sentences")
optparser.add_option("-r", "--reference", dest="reference", default="dev/all.cn-en.en0", help="English reference sentences")
(opts, _) = optparser.parse_args()
ref = [(line.strip().split()) for line in open(opts.reference).readlines()]
ref = ref[:int(opts.length)]
nbests = []

ref = [line.strip().split() for line in open(opts.reference)]
nbests = []

for n, line in enumerate(open(opts.nbest)):
    (i, sentence, features) = line.strip().split("|||")
    (i, sentence) = (int(i), sentence.strip())
    features = [float(h) for h in features.strip().split()]
    if len(ref) <= i:
        break
    while len(nbests) <= i:
        nbests.append([])
    scores = tuple(bleu_stats(sentence.split(), ref[i]))
    bleu_scores = smoothed_bleu(scores)
    inverse_scores = tuple([-x for x in scores])
    nbests[i].append(translation_candidate(sentence, scores, features, bleu_scores))
    if n % 2000 == 0:
        sys.stderr.write(".")

for i in xrange(len(nbests)):
    nbests[i] = sorted(nbests[i], key= lambda h: h.smoothed_bleu)
    max_val = nbests[i][-1]
    nbests[i] = nbests[i][::-1]

num_features = 5
w = [float(1)/5]*num_features
n_epochs = 5
alpha = 0.1
eta = 0.1
n_samples = 100
tau = 5000
for i in xrange(n_epochs):
    for nbest in nbests:
        samples = []
        for idx in xrange(tau):
            rand_indices = numpy.random.randint(len(nbest), size=(2,))
            if nbest[rand_indices[0]].smoothed_bleu < nbest[rand_indices[1]].smoothed_bleu:
                temp = rand_indices[0]
                rand_indices[0] = rand_indices[1]
                rand_indices[1] = temp
            if (nbest[rand_indices[0]].smoothed_bleu) > (alpha + nbest[rand_indices[1]].smoothed_bleu):
                samples.append(tuple((nbest[rand_indices[0]], nbest[rand_indices[1]])))
        samples = sorted(samples, key=lambda h: (-h[0].smoothed_bleu + h[1].smoothed_bleu))
        training_data = samples[:min(len(samples), n_samples)]
        for data_i in training_data:
            if numpy.dot(w, data_i[0].features) <= numpy.dot(w, data_i[1].features):
                w = map(add, w, map(lambda x: x*eta, map(sub, data_i[0].features, data_i[1].features)))


    #features = [float(h) for h in features.strip().split()]

  #w = [1.0/len(features) for _ in xrange(len(features))]

print "\n".join([str(weight) for weight in w])
