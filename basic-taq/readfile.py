from glob import glob
from itertools import (islice, zip_longest)
from collections import Counter
import numpy as np
from collections import Counter
from itertools import islice
import raw_taq_ry

def count_chunk_elements(fname, chunksize=1000000, max_chunk=None, process_chunk=False):

    symbol_roots = Counter()

    for (i,chunk) in enumerate(islice(raw_taq_ry.TAQ2Chunks(fname, 
            	                                             chunksize=chunksize, 
                                                         process_chunk=process_chunk), max_chunk)):

        counts = np.unique(chunk[:]['Symbol_root'], return_counts=True)
        symbol_roots.update(dict(zip_longest(counts[0], counts[1])))

        print("\r {0}".format(i),end="")

    return symbol_roots

if __name__ == '__main__':
    from sys import argv
    fname = '../local_data/EQY_US_ALL_BBO_201402' + argv[1] + '.zip'
    
    chunks = raw_taq_ry.TAQ2Chunks(fname,chunksize=1, process_chunk=False)
    c = count_chunk_elements(fname, max_chunk=None)
    for (i,(k,v)) in enumerate(islice(c.most_common(),10)):
    	print ("\t".join([str(i), k.decode('utf-8').strip(), str(v)]))
