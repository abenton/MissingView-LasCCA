'''
Generate synthetic data as in scalable GCCA paper:

@inproceedings{fu2016efficient,
  title={Efficient and distributed algorithms for large-scale generalized canonical correlations analysis},
  author={Fu, Xiao and Huang, Kejun and Papalexakis, Evangelos E and Song, Hyun-Ah and Talukdar, Partha Pratim and Sidiropoulos, Nicholas D and Faloutsos, Christos and Mitchell, Tom},
  booktitle={Data Mining (ICDM), 2016 IEEE 16th International Conference on},
  pages={871--876},
  year={2016},
  organization={IEEE}
}

Validates the implementation of view-sparse (robust) scalable LasCCA.

Adrian Benton
'''

import numpy as np

import scipy
import scipy.io
import scipy.sparse
import scipy.sparse.linalg

import scalable_gcca

import pandas as pd

from functools import reduce

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
from   matplotlib.backends.backend_pdf import PdfPages

def genSynth(N=1500000, M=100000, V=5, rho=10**(-5), viewSparsity=0.0):
  '''
  See how well we can recover synthetic data with scalable GCCA.  Synthetic data
  generated as described in paper (linked above).
  
  Since all views are perfectly correlated, the optimal correlation captured
  between each pair of views will be $k$, where $k$ are the number of canonical
  directions learned.
  
  Returns training views and handful of views for test.
  
  N: Number of examples
  M: Number of features for each view
  V: Number of views
  rho:          Sparsity of each view
  viewSparsity: Proportion rows that should only contain active features in a single view
  '''
  
  np.random.seed(12345)
  
  Nactive = 5
  
  # Latent features, sparse vectors with five active features in each row,
  # values drawn from a gaussian.
  cols = np.random.randint(M, size=(Nactive * N,))
  rows = np.asarray([j for j in range(N) for i in range(Nactive)])
  data = np.random.randn(*cols.shape) # Positive offset ensures that vanilla LasCCA doesn't accidentally generalize better by pulling projected views towards zero.
  
  Z = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(N, M), dtype=np.float32)
  
  print('Generated Z')
  
  # Generate rows with active features in all views
  As = []
  Xs = []
  
  Ncutoff = int(N * (1.0 - viewSparsity))
  
  for v in range(V):
    # Sparse mapping from latent to observed, values drawn from a gaussian
    A = scipy.sparse.random(M, M, density=rho, format='csr', dtype=np.float32)
    A.data = np.random.randn(*A.data.shape)
    
    As.append( A )
    X = Z[:Ncutoff].dot(A)
    
    X = scipy.sparse.csr_matrix((X.data, X.nonzero()), shape=(N, M))
    
    Xs.append( X )
    print('Generated A_%d -- sparsity %.3e, X_%d -- sparsity %.3e' % (v,
                                                                      len(A.nonzero()[0])/(A.shape[0]*A.shape[1]),
                                                                      v,
                                                                      len(X.nonzero()[0])/(X.shape[0]*X.shape[1])))
  
  # Pick a single view with active features, rest of views are missing data
  if viewSparsity > 0.:
    fullView = np.random.randint(V)
    Xs[fullView][Ncutoff:] = Z[Ncutoff:].dot(As[fullView])
  
  # Test set, 100K examples so we have a good estimate of generalization error
  Xs_test = []
  
  Ntest = 50000
  cols = np.random.randint(M, size=(Nactive * Ntest,))
  rows = np.asarray([j for j in range(Ntest) for i in range(Nactive)])
  data = np.random.randn(*cols.shape)
  
  Ztest = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(Ntest, M), dtype=np.float32)
  
  for v in range(V):
    Xtest = Ztest.dot(As[v])
    
    Xs_test.append( Xtest )
  
  return Xs, Xs_test

def recover(Xs, k, max_epochs=200, findK=False, initGs=None, Xtest=None, Ktest=None):
  '''
  Learns SUMCORR GCCA mappings given data.
  
  Xs:         views, sparse numpy arrays
  k:          number of canonical directions to find
  max_epochs: number of LasCCA iterations
  findK:      should we include a mask of missing views
  initGs:     intialize G
  Xtest:      test data to calculate correlation on
  KTest:      mask for test data
  '''
  
  verbose = False
  
  gcca = scalable_gcca.LasCCA(k, max_epochs, verbose=verbose)
  
  if findK:
    Ks = scalable_gcca.getK(Xs)
  else:
    Ks = [np.ones((X.shape[0], 1), dtype=np.float32)  for X in Xs] # No mask
  
  logs = gcca.learn( Xs, Ks, initGs=initGs, Xtest=Xtest, Ktest=Ktest)
  
  return gcca

def main():
  '''
  Generate sparse synthetic data and learn mappings to correlate views.
  
  Produces plot:
  gcca.synth.vary_sparsity.pdf : Comparison of train/test correlation for robust
                                 vs. vanilla LasCCA.
  '''
  
  # Train correlation captured
  nok_corrs_tr     = []
  subset_corrs_tr  = []
  withk_corrs_tr   = []
  
  # Test correlation captured
  nok_corrs_tst    = []
  subset_corrs_tst = []
  withk_corrs_tst  = []
  
  # Proportion of rows that are missing data from all views but one.
  VIEW_SPARSITY = [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999]
  
  k   = 10
  max_epochs = 5
  V   = 3        # Number of views
  rho = 10**(-1) # Sparsity of latent-to-observed mapping
  N   = 20000    # Number of examples
  M   = 100      # Dimensionality of latent and observed spaces
  
  models = {'robust':[], 'subset':[], 'vanilla':[]}
  
  np.random.seed(12345)
  for viewSparsity in VIEW_SPARSITY:
    print('Generating N=%d, M=%d, Proportion Missing data=%.5f' % (N, M, viewSparsity))
    Xs, Xs_test = genSynth(N=N, M=M, V=V, rho=rho, viewSparsity=viewSparsity)
    Ks, Ks_test = scalable_gcca.getK(Xs), scalable_gcca.getK(Xs_test)
    
    Ncutoff = int(N * (1.0 - viewSparsity))
    
    np.savez_compressed('N-%d_M-%d_V-%d_rho-%e_sparsity-%e.train.npz' %
                        (N, M, V, rho, viewSparsity),
                        **{'row':[X.nonzero()[0] for X in Xs],
                           'col':[X.nonzero()[1] for X in Xs],
                           'value':[X.data for X in Xs],
                           'idToRow':{'example_%d' % (i):i for i in range(N)},
                           'idToCol':{v:{'%d_%d' % (v, i):i for i in range(M)}
                                      for v in range(V)}})
    
    np.savez_compressed('N-%d_M-%d_V-%d_rho-%e_sparsity-%e.test.npz' %
                        (N, M, V, rho, viewSparsity),
                        **{'row':[X.nonzero()[0] for X in Xs_test],
                           'col':[X.nonzero()[1] for X in Xs_test],
                           'value':[X.data for X in Xs_test],
                           'idToRow':{'example_%d' % (i):i for i in range(50000)},
                           'idToCol':{v:{'%d_%d' % (v, i):i for i in range(M)}
                                      for v in range(V)}})
    
    #############
    
    print('Train vanilla...')
    noKGccaModel = recover(Xs=Xs, k=k, max_epochs=max_epochs, findK=False)
    models['vanilla'].append( (viewSparsity, noKGccaModel) )
    
    print('Train vanilla on subset...')
    noKGccaModel_subset = recover(Xs=[X[:Ncutoff] for X in Xs], k=k,
                                  max_epochs=max_epochs,
                                  findK=False)
    models['subset'].append( (viewSparsity, noKGccaModel_subset) )
    
    print('Train robust...')
    withKGccaModel = recover(Xs=Xs, k=k, max_epochs=max_epochs, findK=True)
    models['robust'].append( (viewSparsity, withKGccaModel) )
    
    nok_corrs_tr.append(    noKGccaModel.calcCorr(Xs, Ks)        )
    subset_corrs_tr.append( noKGccaModel_subset.calcCorr(Xs, Ks) )
    withk_corrs_tr.append(  withKGccaModel.calcCorr(Xs, Ks)      )
    
    nok_corrs_tst.append(    noKGccaModel.calcCorr(Xs_test, Ks_test)        )
    subset_corrs_tst.append( noKGccaModel_subset.calcCorr(Xs_test, Ks_test) )
    withk_corrs_tst.append(  withKGccaModel.calcCorr(Xs_test, Ks_test)      )
  
  print('Maximum attainable average correlation between pairs: 1.0')
  print( '-'*16 + 'Proportion Maximum Correlation Captured (K=%d)' % (k) + '-'*16 )
  
  df = pd.DataFrame({'probability_view_missing':VIEW_SPARSITY,
                     'vanilla_train_corr':nok_corrs_tr,
                     'with_mask_train_corr':withk_corrs_tr,
                     'vanilla_test_corr':nok_corrs_tst,
                     'with_mask_test_corr':withk_corrs_tst,
                     'vanilla_subset_train_corr':subset_corrs_tr,
                     'vanilla_subset_test_corr':subset_corrs_tst})
  
  print(df)
  
  # Plot robust vs. vanilla LasCCA correlation captured
  sns.set_style('whitegrid')
  sns.set_context("paper", rc={"font.size":5,"axes.titlesize":8,"axes.labelsize":8})
  fig, ax = plt.subplots()
  x = 1.-df['probability_view_missing']
  vanilla_tr  = df['vanilla_train_corr']
  vanilla_tst = df['vanilla_test_corr']
  robust_tst  = df['with_mask_test_corr']
  robust_tr   = df['with_mask_train_corr']
  
  with PdfPages('gcca.synth.vary_sparsity.pdf') as pdf:
    plt.ylim((0,1.0))
    plt.xscale('symlog', linthreshy=0.01)
    ax.set_xticks(1.- np.asarray(VIEW_SPARSITY))
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter())
    
    plt.plot(x, vanilla_tr, 'r')
    plt.plot(x, robust_tr, 'b')
    
    rl = mlines.Line2D([], [], color='r', linestyle='-',
                       label='LasCCA (Train)')
    bl = mlines.Line2D([], [], color='b', linestyle='-',
                       label='Robust LasCCA (Train)')
    
    plt.xlabel('Proportion Examples with All Views')
    plt.ylabel('Proportion Correlation Captured')
    fig.tight_layout()
    
    plt.legend(handles=[rl, bl], loc='best')
    pdf.savefig()

if __name__ == '__main__':
  main()
