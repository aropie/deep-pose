#!/usr/bin/env python2.6
import numpy as np
import scipy
import scipy.stats.mstats as mstats
import sys

in_data = sys.argv[1:]
list_data = [ float(i) for i in in_data ]
list_data = np.array(list_data)
print '%3.2f'%np.mean(list_data),'+-','%3.2f'%np.std(list_data)
