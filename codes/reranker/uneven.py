import optparse, sys
from collections import namedtuple
from bleu import bleu_stats, smoothed_bleu
from operator import add, abs

optparser = optparse.OptionParser()
translation_candidate = namedtuple("candidate", "sentence, scores, features, smoothed_bleu")
optparser.add_option("-n", "--nbest", dest="nbest", default="train_small.nbest", help="N-best file")
optparser.add_option("-l","--length", dest="length", default=sys.maxint, help="Number of sentences")
optparser.add_option("-r", "--reference", dest="reference", default="dev/all.cn-en.en0", help="English reference sentences")
(opts, _) = optparser.parse_args()
ref = [(line.strip().split()) for line in open(opts.reference).readlines()]
ref = ref[:int(opts.length)]
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
sys.stderr.write("\n")
for i in xrange(len(nbests)):
    nbests[i] = sorted(nbests[i], key=lambda h: h.smoothed_bleu)
    nbests[i] = nbests[i][::-1]

num_features = 5
w = [float(1)/5]*num_features
updates = [0]*num_features
k = 5
r = 5
margin = 0.2
epsilon = 0.00001
converge = 0
while(1):
    for nbest in nbests:
        n = len(nbest)
        if n < (k + r):
            continue
        scores = []
        feats = map(lambda h: h.features, nbest)
        for idx in xrange(len(feats)):
            scores.extend([sum(map(lambda x, y: x*y, feats[idx], w))])
        u = [0.0]*n
        for j in xrange(r):
            for l in xrange(n - k + 1, n):
                if (l - j) > 20:
                    g = 1.0/(j+1) - 1.0/(l+1)
                    if (scores[j] - scores[l]) < g*margin:
                        u[j] += 0.2*g
                        u[l] -= 0.2*g
        w_old = w
        for idx in range(len(feats)):
            feats[idx] = map(lambda x: x * u[idx], feats[idx])
        feats = [sum(x) for x in zip(*feats)]
        w = map(add, feats, w)
        diff = [abs(w_old[i] - w[i]) for i in xrange(len(w))]
        if sum(diff) < epsilon:
            converge = 1
            break
    if converge == 1:
        break

print "\n".join([str(weight) for weight in w])
