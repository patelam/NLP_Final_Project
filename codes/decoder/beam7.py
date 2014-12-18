#!/usr/bin/env python
import optparse
from itertools import groupby
import sys
import models
from operator import itemgetter, add, mul
from collections import namedtuple, Counter

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="dev/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="large/phrase-table/dev-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="lm/en.gigaword.3g.filtered.arpa.gz", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-w", "--weights", dest="weights", default=None)
opts = optparser.parse_args()[0]

if opts.weights is None:
    w = [float(1)/7]*7
else:
    w = []
    for line in open(opts.weights):
        w.extend([float(line)])



tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0, [0.0]*4)]

def getrange(data):
    ranges = []
    for key, group in groupby(enumerate(data), lambda (index, item): index - item):
        group = map(itemgetter(1), group)
        ranges.append(xrange(group[0], group[-1] + 1))
    return ranges

def bitmap(sequence):
    return reduce(lambda x,y: x|y, map(lambda i: long('1'+'0'*i,2), sequence), 0)

def bitmap2str(b, n, on='o', off='.'):
    return '' if n==0 else (on if b&1==1 else off) + bitmap2str(b>>1, n-1, on, off)

def cand_phrases(ranges, f):
    output_phrases = []
    output_idx = []
    for range_i in ranges:
        for j in range_i:
            for k in xrange(j+1, range_i[-1] + 2):
                if f[j:k] in tm:
                    output_phrases.append(tuple(f[j:k]))
                    output_idx.append(xrange(j, k))
    return (output_phrases, output_idx)

def is_ascii(s):
    try:
    	s.decode('ascii')
    except UnicodeDecodeError:
    	return 1
    else:
    	return 0

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for lineid, f in enumerate(french):
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
    coverage_check = bitmap(xrange(len(f)))
    init_bit = 0
    hypothesis = namedtuple("hypothesis", "length, untrans, logprob, lm_prob, tm_prob, lm_state, bit_string, predecessor, start_idx, phrase")
    initial_hypothesis = hypothesis(0.0, 0.0, 0.0, 0.0, [0.0]*4, lm.begin(), 0, None, 0, None)
    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin(), 0] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]: # prune
        #for j in xrange(i+1,len(f)+1):
            cur_str = bitmap2str(coverage_check - h.bit_string, len(f))
            uncovered_idx = []
            for char_id, x in enumerate(cur_str):
                if x == 'o':
                    uncovered_idx.append(char_id)
            ranges = getrange(uncovered_idx)
            phrase = cand_phrases(ranges, f)
            phrase_idx1 = phrase[1]
            phrase = phrase[0]
            length = h.length
            untrans = h.untrans
            logprob = 0
            for phrase_i, phrase_idx in zip(phrase, phrase_idx1):
                j = len(phrase_i) + i
                bit_string = h.bit_string + bitmap(phrase_idx)
                for phrase1 in tm[phrase_i]:
                    tm_prob = map(add, h.tm_prob, phrase1.probs)
                    logprob = h.logprob + sum(map(mul, w[1:5], phrase1.probs))
                    lm_state = h.lm_state
                    for word in phrase1.english.split():
                        logprob += w[6]
                    	length += 1
                    	logprob += is_ascii(word)*w[5]
                    	untrans += is_ascii(word)
                        (lm_state, word_logprob) = lm.score(lm_state, word)
                        lm_prob = h.lm_prob + word_logprob
                        logprob += word_logprob * w[0]
                    lm_prob += lm.end(lm_state) if j == len(f) else 0.0
                    logprob += lm.end(lm_state) if j == len(f) else 0.0
                    new_hypothesis = hypothesis(length, untrans, logprob, lm_prob, tm_prob, lm_state, bit_string, h, phrase_idx[-1] + 1, phrase1)
                    key = (lm_state, bit_string)
                    if key not in stacks[j] or stacks[j][key].logprob < logprob:  # second case is recombination
                        stacks[j][lm_state, bit_string] = new_hypothesis
    final_list = sorted(stacks[-1].itervalues(), key=lambda h: h.logprob)
    winner = final_list[-max(len(final_list), 100):]
    string = []

    def extract_english(h):
        return "" if h.predecessor is None else "%s%s " %(extract_english(h.predecessor), h.phrase.english)
    for num in xrange(len(winner)):
        sentence = extract_english(winner[num])
        eng_words = sentence.strip().split()
        f1 = len(Counter(f) & Counter(eng_words))
        f2 = len(eng_words)
        new_string = (str(lineid) + " ||| " + sentence + " ||| " + str(winner[num].lm_prob) + " ")
        new_string = [new_string]
        new_string.extend([str(p) + " " for p in winner[num].tm_prob])
        new_string.extend([str(float(f1)) + " ", str(float(f2))])
        new_string = ''.join(new_string)
        string.extend([new_string])
    print '\n'.join(string)







