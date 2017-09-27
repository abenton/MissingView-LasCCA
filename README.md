== Robust LasCCA ==

Python implementation of Large-scale GCCA (LasCCA) adapted to support data with no active
features in views.  LasCCA is described here:

> @inproceedings{fu2016efficient,
>   title={Efficient and distributed algorithms for large-scale generalized canonical correlations analysis},
>   author={Fu, Xiao and Huang, Kejun and Papalexakis, Evangelos E and Song, Hyun-Ah and Talukdar, Partha Pratim and Sidiropoulos, Nicholas D and Faloutsos, Christos and Mitchell, Tom},
>   booktitle={Data Mining (ICDM), 2016 IEEE 16th International Conference on},
>   pages={871--876},
>   year={2016},
>   organization={IEEE}
> }

Run synthetic validation experiments with:

    python synth_gcca.py

Sample input data in `sample_data/N-20000_M-100_V-3_rho-1.000000e-01_sparsity-0.000000e+00.train.npz` (compressed numpy array).  Train a model with:

    python scalable_gcca.py --in_path ../sample_data/N-20000_M-100_V-3_rho-1.000000e-01_sparsity-0.000000e+00.train.npz --view 0 1 2 -k 5 --epochs 5 --max_cg_iters 20 --warmstart_cg --prop_heldout 0.0 --model_path ./gcca_retrained_N-20000_M-100_V-3_rho-1.000000e-01_sparsity-0.000000e+00.model.npz --projected_path N-20000_M-100_V-3_rho-1.000000e-01_sparsity-0.000000e+00.projected.npz

Thanks to Xiao Fu for a Matlab reference LasCCA implementation: <http://people.oregonstate.edu/~fuxia/>
