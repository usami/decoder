#!/usr/bin/env python

import sys
import collections
import operator
import math
import optparse
import history

H = history.Feature()

def read_model(fi):
    M = collections.defaultdict(dict)
    L = []
    for line in fi:
        line = line.strip('\n')
        if line.startswith('@label'):
            label = line[7:]
            L.append(label)
        elif line.startswith('@'):
            pass
        else:
            fields = line.split('\t')
            weight = float(fields[0])
            attr = fields[1]
            label = fields[2]
            M[attr][label] = weight

    return M, L

def possible_labels_iob2(L, prev):
    if prev == 'O' or not prev:
        return [x for x in L if not x.startswith('I-')]
    elif prev.startswith('B-'):
        return [x for x in L if (not x.startswith('I-') or x[1:] == prev[1:])]
    elif prev.startswith('I-'):
        return [x for x in L if (not x.startswith('I-') or x[1:] == prev[1:])]
    return L

def possible_labels_iobes(L, prev):
#    if prev == 'O' or not prev:
#        return [x for x in L if x.startswith('B-') or x.startswith('S-') or x == 'O']
#    elif prev.startswith('B-'):
#        return ['I-'+prev[2:], 'L-'+prev[2:]]
#    elif prev.startswith('I-'):
#        return ['I-'+prev[2:], 'L-'+prev[2:]]
#    elif prev.startswith('L-'):
#        return [x for x in L if x.startswith('B-') or x.startswith('S-') or x == 'O']
#    elif prev.startswith('S-'):
#        return [x for x in L if x.startswith('B-') or x.startswith('S-') or x == 'O']
    return L

def logprob(X):
    norm = math.log(sum([math.exp(x) for x in sorted(X.itervalues())]))
    for label in X.iterkeys():
        X[label] -= norm

def score(X, label, M, repl):
    s = 0.
    for x, v in X:
        s += M.get(repl(x), {}).get(label, 0.) * v
    return s

def repl_thru(x):
    return x

class replace1:
    def __init__(self, prev):
        self.prev = prev
    def replace(self, x):
        return x.replace('__$y[-1]', self.prev)

def replace_markov(x, labels, order):
    for i in range(order):
        p = -i-1
        pattern = '__$y[%d]' % p
        dst = labels[p] if i < len(labels) else ''
        x = x.replace(pattern, dst)
    return x

def predict(X, M, L):
    S = {}
    for label in L:
        score = 0.
        token = ''
        for x, v in X:
            score += M.get(x, {}).get(label, 0.) * v
            if x.startswith('w[0]='):
                token = x[5:]
        for x, v in H.get(token, label):
            score += M.get(x, {}).get(label, 0.) * v
        S[label] = score

    S = sorted(S.iteritems(), key=operator.itemgetter(1), reverse=True)
    argmax = S[0][0]
    margin = S[0][1] - S[1][1]
    #argmax = ''
    #smax = None
    #for label, score in S.iteritems():
    #    if smax is None or smax < score:
    #        smax = score
    #        argmax = label
    #margin = 0.
    H.set(token, argmax)
    return argmax, margin

def predict_1st(X, M, L, prev):
    S = {}
    for label in L:
        score = M.get('$y[-1]=' + prev, {}).get(label, 0.)
        token = ''
        for x, v in X:
            score += M.get(x, {}).get(label, 0.) * v
            if x.startswith('w[0]='):
                token = x[5:]
        for x, v in H.get(token, label):
            score += M.get(x, {}).get(label, 0.) * v
        S[label] = score

    S = sorted(S.iteritems(), key=operator.itemgetter(1), reverse=True)
    argmax = S[0][0]
    margin = S[0][1] - S[1][1]
    #argmax = ''
    #smax = None
    #for label, score in S.iteritems():
    #    if smax is None or smax < score:
    #        smax = score
    #        argmax = label
    #margin = 0.
    H.set(token, argmax)
    return argmax, margin


def decode_1st_greedy(seq, M, L, options):
    labels = []
    margins = []
    T = len(seq)
    for t in range(T):
        label, margin = predict_1st(seq[t], M, L, labels[-1] if labels else '')
        labels.append(label)
        margins.append(margin)

    return ['%s\t%f' % (labels[i], margins[i]) for i in range(len(labels))]
    #return labels

def decode_greedy(seq, M, L, options):
    order = 1
    if options.decode == '1st':
        order = 1
    elif options.decode == '2nd':
        order = 2
    elif options.decode == '3rd':
        order = 3

    W = []
    for item in seq:
        for x, v in item:
            if x.startswith('w[0]='):
                W.append(x[5:])

    labels = []
    margins = []
    T = len(seq)
    for t in range(T):
        X = [(replace_markov(x, labels, order), v) for x, v in seq[t]]
        label, margin = predict(X, M, L)
        labels.append(label)
        margins.append(margin)

    return ['%s\t%f' % (labels[i], margins[i]) for i in range(len(labels))]
    #return labels

def decode_1st_viterbi(seq, M, L, options):
    T = len(seq)
    path = [None for t in range(T)]
    table = [{} for t in range(T)]
    links = [{} for t in range(T)]

    if T < 1:
        return []

    # Initialize the scores at t=0
    for label in L:
        table[0][label] = score(seq[0], label, M, lambda x: x)

    # 
    for t in range(1, T):
        for cur in L:
            scores = []
            for prev in L:
                repl = replace1(prev)
                scores.append((prev, score(seq[t], cur, M, repl.replace) + table[t-1][prev]))
            edge = max(scores, key=operator.itemgetter(1))
            links[t][cur] = edge[0]
            table[t][cur] = edge[1]

    t = T-1
    edge = max(table[t].iteritems(), key=operator.itemgetter(1))
    path[t] = edge[0]

    while 0 < t:
        path[t-1] = links[t][path[t]]
        t -= 1

    return path

def decode_1st_viterbi_prob(seq, M, L, options):
    T = len(seq)
    path = [None for t in range(T)]
    table = [{} for t in range(T)]
    links = [{} for t in range(T)]

    if T < 1:
        return []

    # Initialize the scores at t=0
    for label in L:
        table[0][label] = score(seq[0], label, M, lambda x: x)
    logprob(table[0])

    #for label, s in table[0].iteritems():
    #    print label, math.exp(s)

    # 
    for t in range(1, T):
        E = {}
        for prev in L:
            e = {}
            repl = replace1(prev)
            for cur in L:
                e[cur] = score(seq[t], cur, M, repl.replace)
            logprob(e)
            
            #for cur, s in e.iteritems():
            #    print prev, cur, math.exp(s)

            E[prev] = e

        for cur in L:
            scores = []
            for prev in L:
                scores.append((prev, E[prev][cur] + table[t-1][prev]))
            edge = max(scores, key=operator.itemgetter(1))
            links[t][cur] = edge[0]
            table[t][cur] = edge[1]

    t = T-1
    edge = max(table[t].iteritems(), key=operator.itemgetter(1))
    path[t] = edge[0]

    while 0 < t:
        path[t-1] = links[t][path[t]]
        t -= 1

    return path

def token_sequence(seq):
    tokens = []
    for R in seq:
        for f, v in R:
            if f.startswith('w[0]='):
                tokens.append(f[5:])
    return tokens

def sequences(fi):
    seq = []
    lseq = []
    for line in fi:
        line = line.strip('\n')
        if not line:
            yield seq, lseq
            seq = []
            lseq = []
        else:
            fields = line.split('\t')
            lseq.append(fields[0])
            item = [('BIAS', 1.0),]
            for field in fields[1:]:
                item.append((field, 1.))
            seq.append(item)

if __name__ == '__main__':
    fo = sys.stdout

    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--decode', dest='decode', default='1st',
        help='specify the decoder'
        )
    parser.add_option(
        '-f', '--feature', dest='feature', action='append', default=[],
        help='configure feature set'
        )

    (options, args) = parser.parse_args()

    (M, L) = read_model(open(args[0]))

    decode = None
    if options.decode == '1st':
        decode = decode_1st_greedy
    elif options.decode == '2nd':
        decode = decode_greedy
    elif options.decode == '3rd':
        decode = decode_greedy
    elif options.decode == '1st-viterbi':
        decode = decode_1st_viterbi_prob
    else:
        fe.write('ERROR: decoder not found\n')
        sys.exit(1)

    for seq, lseq in sequences(sys.stdin):
        vseq = decode(seq, M, L, options)
        tseq = token_sequence(seq)
        for i in range(len(lseq)):
            #fo.write('%s\t%s\t%s\n' % (tseq[i], lseq[i], vseq[i]))
            fo.write('%s\t%s\n' % (lseq[i], vseq[i]))
        fo.write('\n')

