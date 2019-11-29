Here are some implementations of EVM and ESVM in order to provide reproducibility.

First, to run experiments on the SUSY dataset, please download it via the link
https://drive.google.com/file/d/1XApFlis670clXSirjmPiXoPGAM7GE4Os/view?usp=sharing
and put the downloaded file into the data directory.

To reproduce all our experiments, simply run through corresponding jupyter notebook. 
For results included in the main part of the paper, you may want to adjust parameter f_type:
-f_type == "posterior_prob_mean" to run experiments for the average predictive distribution;
-f_type == "posterior_prob_point" to run experiments for the predictive distribution of a fixed test point;