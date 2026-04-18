import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Allow running this script from the homework subfolder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils


def parse_position(mut):
    if mut == "wt":
        return np.nan
    m = re.match(r"^[A-Z](\d+)[A-Z]$", mut)
    if m is None:
        return np.nan
    return int(m.group(1))


def load_method_df(evol_indices_dir, protein_name, method, num_samples):
    if method == "msa_direct":
        filename = f"{protein_name}_1_samples_msa_direct.csv"
    else:
        filename = f"{protein_name}_{num_samples}_samples_{method}.csv"
    path = os.path.join(evol_indices_dir, filename)
    df = pd.read_csv(path)
    return df, path


def compute_auc(df_scores, labels_df):
    merged = pd.merge(
        df_scores[["protein_name", "mutations", "evol_indices"]],
        labels_df[["protein_name", "mutations", "ClinVar_labels"]],
        how="inner",
        on=["protein_name", "mutations"],
    )
    if len(merged) == 0 or merged["ClinVar_labels"].nunique() < 2:
        return np.nan, len(merged)
    auc = roc_auc_score(merged["ClinVar_labels"], merged["evol_indices"])
    return float(auc), len(merged)


def compute_pssm_conservation(msa_path, theta):
    data = data_utils.MSA_processing(
        MSA_location=msa_path,
        theta=theta,
        use_weights=True,
        weights_location=f"./data/weights/{os.path.basename(msa_path)}_theta_{theta}.npy",
    )

    weighted_counts = (data.one_hot_encoding * data.weights[:, None, None]).sum(axis=0) + 1.0
    weighted_freqs = weighted_counts / weighted_counts.sum(axis=1, keepdims=True)

    entropy = -np.sum(weighted_freqs * np.log(weighted_freqs + 1e-12), axis=1)
    pssm_conservation = 1.0 - entropy / np.log(data.alphabet_size)

    records = []
    for focus_idx, idx_col in enumerate(data.focus_cols):
        uniprot_pos = idx_col + data.focus_start_loc
        records.append({
            "position": int(uniprot_pos),
            "pssm_conservation": float(pssm_conservation[focus_idx]),
        })
    return pd.DataFrame(records)


def compute_eve_position_conservation(df_scores):
    temp = df_scores.copy()
    temp = temp[temp["mutations"] != "wt"].copy()
    temp["position"] = temp["mutations"].map(parse_position)
    temp = temp.dropna(subset=["position"])
    temp["position"] = temp["position"].astype(int)
    grouped = temp.groupby("position", as_index=False)["evol_indices"].mean()
    grouped = grouped.rename(columns={"evol_indices": "eve_position_conservation"})
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_file", required=True)
    parser.add_argument("--labels_file", required=True)
    parser.add_argument("--evol_indices_dir", required=True)
    parser.add_argument("--num_samples", required=True, type=int)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mapping = pd.read_csv(args.mapping_file)
    protein_name = mapping.loc[0, "protein_name"]
    msa_file = mapping.loc[0, "msa_location"]
    theta = float(mapping.loc[0, "theta"])

    labels_df = pd.read_csv(args.labels_file)
    labels_df = labels_df[labels_df["protein_name"] == protein_name].copy()

    methods = ["bayes_z10", "bayes_z50", "bayes_z100", "nonbayes_z50", "msa_direct"]

    auc_rows = []
    loaded = {}
    for method in methods:
        df_method, path = load_method_df(args.evol_indices_dir, protein_name, method, args.num_samples)
        loaded[method] = df_method
        auc, n_overlap = compute_auc(df_method, labels_df)
        auc_rows.append(
            {
                "method": method,
                "auc": auc,
                "n_labeled_overlap": n_overlap,
                "source_file": path,
            }
        )

    auc_df = pd.DataFrame(auc_rows).sort_values(by="auc", ascending=False)
    auc_path = os.path.join(args.output_dir, "auc_summary.csv")
    auc_df.to_csv(auc_path, index=False)

    baseline_eve = loaded["bayes_z50"]
    eve_pos = compute_eve_position_conservation(baseline_eve)

    msa_path = os.path.join("./data/MSA", msa_file)
    pssm_pos = compute_pssm_conservation(msa_path, theta)

    conservation = pd.merge(eve_pos, pssm_pos, on="position", how="inner")
    eve_min = conservation["eve_position_conservation"].min()
    eve_max = conservation["eve_position_conservation"].max()
    conservation["eve_position_conservation_norm"] = (
        conservation["eve_position_conservation"] - eve_min
    ) / (eve_max - eve_min + 1e-12)
    conservation["abs_diff"] = np.abs(
        conservation["eve_position_conservation_norm"] - conservation["pssm_conservation"]
    )

    pearson = conservation["eve_position_conservation"].corr(conservation["pssm_conservation"], method="pearson")
    spearman = conservation["eve_position_conservation"].corr(conservation["pssm_conservation"], method="spearman")

    conservation_path = os.path.join(args.output_dir, "conservation_comparison.csv")
    conservation.to_csv(conservation_path, index=False)

    top_eve = conservation.nlargest(10, "eve_position_conservation")[["position", "eve_position_conservation"]]
    top_pssm = conservation.nlargest(10, "pssm_conservation")[["position", "pssm_conservation"]]
    top_diff = conservation.nlargest(10, "abs_diff")[["position", "eve_position_conservation", "eve_position_conservation_norm", "pssm_conservation", "abs_diff"]]

    top_eve_path = os.path.join(args.output_dir, "top_eve_conserved_positions.csv")
    top_pssm_path = os.path.join(args.output_dir, "top_pssm_conserved_positions.csv")
    top_diff_path = os.path.join(args.output_dir, "top_conservation_disagreement_positions.csv")
    top_eve.to_csv(top_eve_path, index=False)
    top_pssm.to_csv(top_pssm_path, index=False)
    top_diff.to_csv(top_diff_path, index=False)

    auc_lookup = {row["method"]: row["auc"] for _, row in auc_df.iterrows()}

    report_path = os.path.join(args.output_dir, "homework_report.md")
    with open(report_path, "w") as f:
        f.write("# EVE Homework Report\n\n")
        f.write(f"Protein: {protein_name}\n\n")
        f.write("## 1) EVE run status\n")
        f.write("Completed end-to-end baseline run (train VAE, compute evol indices, train/load GMM, output EVE scores and plots).\n\n")

        f.write("## 2) Latent dimension comparison\n")
        f.write("- Bayesian z=10 AUC: {:.4f}\n".format(auc_lookup.get("bayes_z10", np.nan)))
        f.write("- Bayesian z=50 AUC: {:.4f}\n".format(auc_lookup.get("bayes_z50", np.nan)))
        f.write("- Bayesian z=100 AUC: {:.4f}\n\n".format(auc_lookup.get("bayes_z100", np.nan)))

        f.write("## 3) PSSM vs EVE residue conservation\n")
        f.write("- Pearson correlation: {:.4f}\n".format(float(pearson)))
        f.write("- Spearman correlation: {:.4f}\n".format(float(spearman)))
        f.write("- Detailed table: conservation_comparison.csv\n\n")

        f.write("## 4) Direct MSA input vs EVE\n")
        f.write("- MSA direct score AUC: {:.4f}\n".format(auc_lookup.get("msa_direct", np.nan)))
        f.write("- EVE Bayesian z=50 AUC: {:.4f}\n\n".format(auc_lookup.get("bayes_z50", np.nan)))

        f.write("## 5) Bayesian decoder vs non-Bayesian decoder\n")
        f.write("- Bayesian z=50 AUC: {:.4f}\n".format(auc_lookup.get("bayes_z50", np.nan)))
        f.write("- Non-Bayesian z=50 AUC: {:.4f}\n\n".format(auc_lookup.get("nonbayes_z50", np.nan)))

        f.write("## Output files\n")
        f.write("- auc_summary.csv\n")
        f.write("- conservation_comparison.csv\n")
        f.write("- top_eve_conserved_positions.csv\n")
        f.write("- top_pssm_conserved_positions.csv\n")
        f.write("- top_conservation_disagreement_positions.csv\n")

    print("Saved:")
    print(auc_path)
    print(conservation_path)
    print(report_path)


if __name__ == "__main__":
    main()
