    #!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
import math

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="dev/all.cn-en", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en0", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="cn", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training with Dice's coefficient...")

#F given E
target_files = ['']
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))]
vocab_count = defaultdict(int)


for (f, e) in bitext:
    e.extend(' ')

for (f, e) in bitext:
    for f_i in set(f):
        vocab_count[f_i] +=1

k=0
iterations=0
weights = defaultdict(float)
for (f, e) in bitext:
    for (i,f_i) in enumerate(f):
        for (j,e_j) in enumerate(e):
            if(f_i == e_j):
                weights[(f_i,e_j)]=float(5.0/len(vocab_count))
            elif(i==j):
                weights[(f_i,e_j)]=float(3.0/len(vocab_count))
            else:
                weights[(f_i,e_j)]=float(1.0/len(vocab_count))
while(k<=5):
    sys.stderr.write(".")
    k+=1
    e_count = defaultdict(int)
    fe_count = defaultdict(int)
    for (f, e) in bitext:
        for f_i in set(f):
            norm_z=0.0
            for e_j in set(e):
                norm_z+= weights[(f_i,e_j)]
            for e_j in set(e):
                count_temp = float((weights[(f_i,e_j)]) / float(norm_z))
                if e_j != ' ':
                    fe_count[(f_i,e_j)] += count_temp
                    e_count[e_j] += count_temp
                else:
                    fe_count[(f_i,e_j)] += count_temp/2
                    e_count[e_j] += count_temp/2
    for (f, e) in fe_count:
        weights[(f,e,k)]=fe_count[(f, e)] / e_count[e]

for fe in weights:
    f = fe[0]
    e = fe[1]
    print "%s\t%s %s"%(f,e,weights[fe])
