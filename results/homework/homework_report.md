# EVE Homework Report

Protein: P53_HUMAN

## 1) EVE run status
Completed end-to-end baseline run (train VAE, compute evol indices, train/load GMM, output EVE scores and plots).

## 2) Latent dimension comparison
- Bayesian z=10 AUC: 0.8493
- Bayesian z=50 AUC: 0.9470
- Bayesian z=100 AUC: 0.9154

## 3) PSSM vs EVE residue conservation
- Pearson correlation: 0.4880
- Spearman correlation: 0.6119
- Detailed table: conservation_comparison.csv

## 4) Direct MSA input vs EVE
- MSA direct score AUC: 0.9818
- EVE Bayesian z=50 AUC: 0.9470

## 5) Bayesian decoder vs non-Bayesian decoder
- Bayesian z=50 AUC: 0.9470
- Non-Bayesian z=50 AUC: 0.9279

## Output files
- auc_summary.csv
- conservation_comparison.csv
- top_eve_conserved_positions.csv
- top_pssm_conserved_positions.csv
- top_conservation_disagreement_positions.csv
