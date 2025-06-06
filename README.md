# Jacobian Alignment Explains Grokking and Centroid Alignment Identifies It

[Paper]() | [Blog](https://thomaswalker1.github.io/blog/centroid_alignment_grokking.html)

This is the official GitHub repository for our paper "Jacobian Alignment Explain Grokking and Centroid Alignment Identifies It". `utils.py` provides classes to perform Jacobian regularisation, compute centroids, and compute the explained variance of the first principal component of the Jacobian matrix on a model for input batches.

- `comparison_to_ntk.py`: Code for the experiment of Section 4.1, with the default configuration values reproducing the data for Figure 3.
- `xor_grokking.py`: Code for the experiment of Section 4.2, with the default configuration values reproducing the data for the top row of Figure 4, and the dafault values with `jr=1.0` reproducing the data for the bottom row of Figure 4.
- `delayed_robustness.py`: Code for the experiment of para "Inducing Delayed Robustness." of Section 4.3, with the default configuration values reproducing the data for the "Weight-Decay" run and the default configuration values with `jr=0.001` reproducing the data for the "Jacobian Regualrisation" run of Figure 5.
- `inhibiting_generalisation.py`: Code for the experiment of para "Inhibiting Delayed Generalisation." of Section 4.3, with the defauly configuration values reproducing the data for the "Minimised Jacobian" run and the defauly configuration values with `jr=5.0` reproducing the data for "Constrained Jacobian" run of Figure 6.
- `accelerating_grokking.py`: Code for the experiment of para "Accelerating Grokking." of Section 4.3, with the default configuration values reproducing the "Baseline": setting `jr=1e-3`,`loss_fn='CrossEntropy` reproduces the "Jacobian Regularisation" run under the cross-entropy loss function, setting `jr=1e-4`,`loss_fn='MSE` reproduces the "Jacobian Regularisation" run under the mean-squared error loss function, utilising `--grokfast` reproduces the Grokfast run, and utilising `-adv_training` reproduces the "Adversarial Training" run.

## Citation

    @article{,
      title={Jacobian Alignment Explains Grokking and Centroid Alignment Identifies It},
      author={Walker, Thomas and Humayun, Ahmed Imtiaz and Balestriero, Randall and Baraniuk, Richard},
      journal={arXiv},
      year={2025}
    }