'''
Implementation of scalable GCCA as described in:

@inproceedings{fu2016efficient,
  title={Efficient and distributed algorithms for large-scale generalized canonical correlations analysis},
  author={Fu, Xiao and Huang, Kejun and Papalexakis, Evangelos E and Song, Hyun-Ah and Talukdar, Partha Pratim and Sidiropoulos, Nicholas D and Faloutsos, Christos and Mitchell, Tom},
  booktitle={Data Mining (ICDM), 2016 IEEE 16th International Conference on},
  pages={871--876},
  year={2016},
  organization={IEEE}
}

that supports sparse views (subset of examples have views without any active features).

Adrian Benton
'''

import argparse, os, time

import h5py
import numpy as np

import scipy
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg

from scipy.sparse.linalg.interface import LinearOperator
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import non_reentrant

import ctypes as c
from multiprocessing import Process
from multiprocessing import Array

from functools import reduce

REG_EPSILON = 1.e-6

def loadNpzData(npzPath, views, onlyNonzeroRows=False):
  '''
  Load numpy arrays, already split into views. 
  '''
  
  d = np.load(npzPath)

  # Column and row indices indexed by view
  cs = d['col']
  rs = d['row']
  vs = d['value']
  
  idToRow = d['idToRow'].item() # Example labels
  idToCol = d['idToCol'].item() # View ID to feature label to feature index
  
  colToIds = []
  rowToId  = { r:rid for rid, r in idToRow.items() }
  
  Xs = []
  
  maxRow = max([max(r) for r in rs])
  totalCols = 0
  
  uniqRows = []
  
  for view in views:
    viewIdx = view
    maxCol  = max(cs[viewIdx])
    totalCols += (maxCol+1)
    
    Xs.append( scipy.sparse.csr_matrix( ( vs[viewIdx],
                                         (rs[viewIdx], cs[viewIdx]) ),
                                        shape=(maxRow+1, maxCol+1), dtype=np.float32 ) )
    
    colToIds.append( {v:k for k,v in idToCol[view].items()} )
    
    uniqRows.append( np.unique(rs[viewIdx]) )
    
    print('Loaded view:', view)
  
  # Only include examples with all views active
  if onlyNonzeroRows:
    keptRows = sorted(list(reduce(lambda x,y:x&y, [set(row) for row in uniqRows])))
    
    Xs = [X[keptRows,:] for X in Xs]
    
    rowToId = {i:rowToId[keptRowIdx] for i, keptRowIdx in enumerate(keptRows)}
  
  totalActive = sum([X.nnz for X in Xs])
  print('Loaded data:', Xs[0].shape[0], totalCols, totalActive)
  
  return Xs, rowToId, colToIds

def vector_cg(AtA, AtG, X, cols, X0=None, maxiter=20):
  ''' Solve a subset of linear CG problems in series. '''
  for col in cols:
    if X0 is not None:
      X[:,col], convergence_info = scipy.sparse.linalg.cg(AtA, AtG[:,col],
                                                          x0=X0[:,col],
                                                          maxiter=maxiter)
    else:
      X[:,col], convergence_info = scipy.sparse.linalg.cg(AtA, AtG[:,col],
                                                          maxiter=maxiter)

def matrix_cg(A, G, K, X0=None, maxiter=20, print_resid=True, AtA=None, num_threads=20):
  '''
  Solves a least-squares problem for each column of G by linear CG.  Parallelized
  over all columns.
  '''
  
  start = time.time()
  
  num_cols = G.shape[1]

  nz_idx = np.where(K[:,0])
  Aprime = scipy.sparse.diags(K[:,0]).dot( A )
  Aprime.eliminate_zeros()
  Gprime = K * G
  
  mp_arr = Array(c.c_float, Aprime.shape[1]*num_cols) # shared by multiple processes
  arr    = np.frombuffer(mp_arr.get_obj(), dtype=np.float32)
  X      = arr.reshape((Aprime.shape[1],num_cols)) # b and arr share the same memory
  
  AtG = Aprime.T.dot(Gprime)
  
  if AtA is None:
    AtA = Aprime.T.dot(Aprime)
  
  # Parallelize this across threads
  all_cols = [[] for i in range(num_threads)]
  procs = [Process(target=vector_cg, args=(AtA, AtG, X,
                                           [c for ci, c in enumerate(range(num_cols))
                                            if (ci % num_threads) == i], X0, maxiter))
           for i in range(num_threads)]
  
  for p in procs:
    p.start()
  for p in procs:
    p.join()
  
  end = time.time()
  
  if print_resid:
    K = scipy.sparse.diags(K[:,0])
    K_true = scipy.sparse.diags( 1. * (np.asarray( (A.power(2)).sum(axis=1) ) > 0.)[:,0] ).dot( scipy.sparse.diags( 1. * ((G**2.).sum(axis=1) > 0.) ) )
    
    r = (A).dot(X) - G # residual over all examples
    r_only_nz =  K.dot( r ) # residual over only nonzero
    r_only_true_nz =  K_true.dot( r ) # residual over only nonzero
    
    na, nb, nr, nr_only_nz, nb_only_nz = np.linalg.norm( X ), np.linalg.norm( G ), np.linalg.norm( r ), np.linalg.norm( r_only_nz ), np.linalg.norm( K.dot( G ) )
    nr_only_true_nz, nb_only_true_nz = np.linalg.norm( r_only_true_nz ), np.linalg.norm( K_true.dot( G ) )
    
    print('|A|=%.3e, |B|=%.3e, |A X - B|=%.3e, |K ( A X - B )|=%.3e, |K_{true} ( A X - B )|=%.3e, norm_resid=%.3f, norm_resid_nz=%.3f, norm_resid_true_nz=%.3f' % (na, nb, nr, nr_only_nz,
                                                                                                                                                                   nr_only_true_nz,
                                                                                                                                                                   nr/nb, nr_only_nz/nb_only_nz,
                                                                                                                                                                   nr_only_true_nz/nb_only_true_nz))
  
  return X

class LasCCA:
  ''' Reimplementation of LasCCA supporting missing views. '''
  
  def __init__(self, k, max_epochs=100, verbose=True):
    self.k = k
    self.max_epochs = max_epochs
    
    self.max_cg_iter = 20
    self.verbose = verbose
    
    self.Qs = None
    self.P  = None
  
  def learn( self, Xs, Ks, modelPath=None, initGs=None, Xtest=None, Ktest=None):
    '''
    Learn canonical weights by LasCCA.
    
    Xs:        [ sparse numpy array], training views
    Ks:        [ binary numpy vectors ], masks over which examples are missing data
    modelPath: if not None, writes model to disk if objective improved
    initGs:    [ {n \times k} numpy array ], initialization for G (random init if None)
    Xtest:     [ sparse numpy array], test views
    Ktest:     [ binary numpy vectors ], masks for test data
    '''
    
    K_sums    = [ sum([Kj for j, Kj in enumerate(Ks) if j != i])/(len(Ks)-1)
                  for i, Ki in enumerate(Ks) ] # partial sums over K, excluding K_i
    
    invK_sums = [] # multiplicative inverse of partial sums
    for Ksum in K_sums:
      inv_Ksum = 1./Ksum
      inv_Ksum[np.isnan(inv_Ksum)] = 0.
      inv_Ksum[np.isinf(inv_Ksum)] = 0. # divide by zeroes
      invK_sums.append( inv_Ksum )
    
    # Mask for feature active in view i, and feature active in at least one other view
    K_sums_and_i = [1.* ( (invKSum > 0.) & (Ki > 0.) )
                    for invKSum, Ki in zip(invK_sums, Ks)]
    
    XtXs = [X.T.dot(X) for X in Xs]
    
    n = Xs[0].shape[0]
    V = len(Xs)
    
    # Keep track of objective each epoch
    obj_0    = 0.0
    max_obj  = obj_0
    obj_00   = np.zeros((V,), dtype=np.float32)
    obj_000  = np.zeros((V, self.max_epochs), dtype=np.float32)
    time_000 = np.zeros((V, self.max_epochs), dtype=np.float32)
    time_1   = np.zeros((self.max_epochs), dtype=np.float32)
    
    self.Qs = [ np.zeros((X.shape[1], self.k), dtype=np.float32) for X in Xs ]
    
    # initialize G by orthogonalizing a random matrix
    # Make sure to normalize only over those examples that have at least one
    # other active view and are active themselves
    if initGs is None:
      Gs = []
      for i in range(V):
        Gi = np.random.randn(n, self.k).astype(np.float32)
        Ui, Si, Vti = scipy.linalg.svd(Gi, full_matrices=False,
                                       compute_uv=True,
                                       overwrite_a=True,
                                       check_finite=False)
        Gi = (K_sums_and_i[i] * Ui / np.linalg.norm(K_sums_and_i[i] * Ui,
                                                    axis=0,
                                                    keepdims=True) ).dot(Vti)
        Gs.append(Gi)
    else:
      Gs = initGs
    
    if self.verbose:
      print('init G')
    
    for j in range(V):
      PP = np.zeros((n, self.k), dtype=np.float32)
      for i in range(V):
        if i != j:
          PP = PP + Gs[i]

     # Don't want to count views with no active features
      obj_00[j] = np.trace((Ks[j] * Gs[j]).T.dot( invK_sums[j] * PP )) / (Ks[j] * invK_sums[j] ).sum()
    
    obj_0 = obj_00.sum()
    
    if self.verbose:
      print('init obj', obj_0)
    
    XpinG    = [ None for i in range(V) ]
    XtXXpinG = [ None for i in range(V) ]
    Mi_old   = [ None for i in range(V) ]
    Mi       = [ None for i in range(V) ]
    
    corr_history = []
    
    for it in range(self.max_epochs):
      start = time.time()
      
      # Shared state for all views
      P = np.zeros((n, self.k), dtype=np.float32)
      
      for i in range(V):
        if self.verbose:
          print('XpinG[%d]' % (i), end=' -- ')
        
        XpinG[i]    = matrix_cg( Xs[i], Gs[i], K=K_sums_and_i[i],
                                 X0=XpinG[i], maxiter=20, AtA=None,
                                 print_resid=self.verbose )
        XtXXpinG[i] = Xs[i].dot( XpinG[i] )
        P = P + XtXXpinG[i]
      time_1[it] = time.time() - start
      
      for i in range(V):
        start = time.time()
        
        Pi = P - XtXXpinG[i]
        
        if self.verbose:
          print('Mi_old[%d]' % (i), end=' -- ')
        
        # Find mapping of X_i to sum of projections from other views
        Mi_old[i] = matrix_cg( Xs[i], invK_sums[i] * Pi, K=K_sums_and_i[i],
                               X0=Mi_old[i], maxiter=20, AtA=None,
                               print_resid=self.verbose )
        
        Mi[i] = K_sums_and_i[i] * (Xs[i].dot(Mi_old[i]))
        
        # Whiten
        Ui, Si, Vti = scipy.linalg.svd( Mi[i] , full_matrices=False,
                                        compute_uv=True, overwrite_a=False,
                                        check_finite=False)
        Gs[i] = Ui.dot(Vti)
        
        if self.verbose:
          print('Qs[%d]' % (i), end=' -- ')
        
        # Find mapping of X_i to G_i
        self.Qs[i]  = matrix_cg( Xs[i], Gs[i], K=K_sums_and_i[i], X0=self.Qs[i],
                                 maxiter=20, AtA=None, print_resid=self.verbose )
        
        P = Pi + Xs[i].dot( self.Qs[i] )
        
        obj_00[i] = ( np.trace( (Ks[i] * Gs[i]).T.dot( (invK_sums[i] * Mi[i]) ) ) /
                      ( Ks[j] * invK_sums[j] ).sum() )
        obj_000[i,it] = obj_00.sum()
        time_000[i,it] = time.time() - start
        
        if self.verbose:
          print( 'Update block', i, it, time_000[i,it], obj_00[i] )
        
        if (Xtest is not None) and (Ktest is not None):
          corr_history.append( (it, i, self.calcCorr(Xtest, Ktest) ) )
          print( it, corr_history[-1][2] )
      
      obj_0 = obj_00.sum()
      max_obj = max([obj_0, max_obj])
      print( 'finished iter', it )
      
      self.P = P
      
      if (modelPath is not None) and (obj_0 == max_obj):
        self.serialize(modelPath)
        
        print('CorrSum Objective=%s, Saved model' % (max_obj))
    
    return corr_history
  
  def apply(self, Xs):
    ''' Project heldout views. '''
    Xs_proj = [ X.dot(Q) for X, Q in zip(Xs, self.Qs) ]
    
    return Xs_proj
  
  def calcPairCorrs(self, Xs, Ks):
    '''
    Calculate correlation between pairs.  View pairs where data is missing
    from either view are ignored when calculating this.
    '''
    
    Xs_proj = self.apply(Xs)
    
    corrs = []
    
    for i, Xi in enumerate(Xs_proj):
      for j, Xj in enumerate(Xs_proj[(1+i):]):
        Ki, Kj = Ks[i], Ks[i+j+1]
        Kij = scipy.sparse.diags( (Ki*Kj)[:,0] ) # must have active features in both views
        X1 = Kij * Xi
        X2 = Kij * Xj
        
        if (np.abs(X1).sum() > 0.) and (np.abs(X2).sum() > 0.):
          # Unit-normalize columns after removing missing rows
          X1 = X1 / np.linalg.norm(X1, axis=0, keepdims=True)
          X2 = X2 / np.linalg.norm(X2, axis=0, keepdims=True)
          
          corrs.append( (i, i+j+1, X1.T.dot(X2).trace() ) )
        else:
          corrs.append( (i, i+j+1, float('-inf') ) )
    
    return corrs
  
  def calcCorr(self, Xs, Ks):
    '''
    Calculate average correlation between pairs on a heldout set.
    K ensures that views with missing data are ignored.
    '''
    
    corrs = self.calcPairCorrs( Xs, Ks )
    valid_corrs = [corr for i,j,corr in corrs if not np.isinf(corr)]
    
    total_corr  = np.sum( valid_corrs )
    
    V = len(Xs)
    
    # average correlation per component captured between pair of views
    avg_corr = total_corr / ( len(valid_corrs) * self.k )
    
    return avg_corr
  
  @staticmethod
  def deserialize(path):
    '''
    Load saved model
    '''
    d = np.load(path)
    
    model = LasCCA(d['k'], d['max_epochs'])
    
    model.Qs = model.Qs
    model.P = model.P
    
    return model
  
  def serialize(self, path):
    '''
    Write model to disk
    '''
    np.savez_compressed(path,
                        **{'Qs':self.Qs, 'P':self.P,
                           'k':self.k, 'max_epochs':self.max_epochs})

def getK(Xs):
  ''' Find K, a mask over entries to only pick out views with some active features. '''
  Ks = [1.*np.asarray(X.sum(axis=1)!=0.) for X in Xs]
  
  return Ks

def learn(Xs, k, epochs, modelPath=None, prop_heldout=0.0, continue_training=False, findK=False, verbose=False):
  ''' Convenience method to train LasCCA. '''
  
  # Make sure Xs are all the same length
  num_users = max([X.shape[0] for X in Xs])
  
  # Binary mask over GL rows
  if findK:
    Ks = getK(Xs)
  else:
    Ks = [np.ones((X.shape[0], 1), dtype=np.float32)  for X in Xs] # No mask
  
  if prop_heldout > 0.0:
    splitIdx = int(Xs[0].shape[0]*(1. - prop_heldout))
    
    Xs_train = [X[:splitIdx] for X in Xs]
    Xs_test  = [X[splitIdx:] for X in Xs]
    
    Ks_train = [K[:splitIdx] for K in Ks]
    Ks_test  = [K[splitIdx:] for K in Ks]
  else:
    Xs_train = Xs
    Xs_test  = None
    
    Ks_train = Ks
    Ks_test  = None
  
  if not continue_training:
    gcca = LasCCA(k, max_epochs=epochs, verbose=verbose)
  else:
    print('Loading old model!')
    gcca = LasCCA.deserialize(modelPath)
    gcca.verbose = verbose
  
  gcca.learn(Xs_train, Ks_train, modelPath)
  
  corr_train = gcca.calcPairCorrs(Xs_train, Ks_train)
  print('Train Corr per pair: %s (%s)' % (corr_train,
                                          gcca.calcCorr(Xs_train, Ks_train)))
  
  if prop_heldout > 0.0:
    corr_test = gcca.calcPairCorrs(Xs_test, Ks_test)
    print('Test Corr per pair: %s (%s)' % (corr_test,
                                           gcca.calcCorr(Xs_test, Ks_test)))
  
  return gcca

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--in_path", '-i', required=True, default=None,
                      help='Preprocessed npz file')
  parser.add_argument("--views", required=True, default=[], nargs='+', type=int,
                      help='Which views to select')
  parser.add_argument("-k", required=True, type=int,
                      help='Number of canonical directions to learn')
  parser.add_argument("--epochs", '-e', default=100, type=int,
                      help='Number of BCD passes over the views')
  parser.add_argument("--max_cg_iters", default=20, type=int,
                      help='Number of iterations for the linear CG subroutines')
  parser.add_argument("--prop_heldout", default=0.2, type=float,
                      help='Proportion of heldout rows')
  
  parser.add_argument("--verbose", action='store_true', default=False,
                      help='Verbose training')
  parser.add_argument("--warmstart_cg", action='store_true', default=False,
                      help='whether to warm-start the linear CG routines')
  
  parser.add_argument("--model_path", default=None,
                      help='Where to save learned model')
  parser.add_argument("--projected_path", '-o', default=None,
                      help='Where to save projected views')
  
  parser.add_argument("--continue_training", action='store_true', default=False,
          help='If model path already exists, continue training where we left off')
  parser.add_argument("--only_nonzero_rows", action='store_true', default=False,
          help='only keep rows with features active in all views')
  args = parser.parse_args()
  
  Xs, rowToId, colToIds = loadNpzData(args.in_path, args.views,
                                      onlyNonzeroRows=args.only_nonzero_rows)
  
  model = learn(Xs, args.k, args.epochs, args.model_path,
                args.prop_heldout, args.continue_training, findK=True,
                verbose=args.verbose)
