#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MSA_DATA_FOLDER="./data/MSA"
MSA_LIST="./data/mappings/homework_p53_mapping.csv"
PROTEIN_INDEX=0
MSA_WEIGHTS_LOCATION="./data/weights"
VAE_CHECKPOINT_LOCATION="./results/VAE_parameters"
TRAINING_LOGS_LOCATION="./logs"
MODEL_PARAMETERS_DEFAULT="./EVE/default_model_params.json"
ALL_SINGLES_MUTATIONS_FOLDER="./data/mutations"
OUTPUT_EVOL_INDICES_LOCATION="./results/evol_indices"
OUTPUT_EVE_SCORES_LOCATION="./results/EVE_scores"
LABELS_FILE="./data/labels/ClinVar_labels_P53_PTEN_RASH_SCN5A.csv"
PLOT_LOCATION="./results"

HOMEWORK_ROOT="./results/homework"
PARAM_DIR="${HOMEWORK_ROOT}/params"
mkdir -p "${HOMEWORK_ROOT}" "${PARAM_DIR}" "${ALL_SINGLES_MUTATIONS_FOLDER}" "${OUTPUT_EVOL_INDICES_LOCATION}" "${OUTPUT_EVE_SCORES_LOCATION}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-2500}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SEED="${SEED:-42}"

create_param_file () {
  local z_dim="$1"
  local bayesian="$2"
  local output_json="$3"
  python - <<PY
import json
src = "${MODEL_PARAMETERS_DEFAULT}"
out = "${output_json}"
with open(src, "r") as f:
    params = json.load(f)
params["encoder_parameters"]["z_dim"] = int(${z_dim})
params["decoder_parameters"]["z_dim"] = int(${z_dim})
params["decoder_parameters"]["bayesian_decoder"] = ${bayesian}
params["training_parameters"]["num_training_steps"] = int(${NUM_TRAIN_STEPS})
params["training_parameters"]["save_model_params_freq"] = int(${NUM_TRAIN_STEPS}) + 1
params["training_parameters"]["log_training_freq"] = max(100, int(${NUM_TRAIN_STEPS}) // 5)
with open(out, "w") as f:
    json.dump(params, f, indent=2)
print("Wrote", out)
PY
}

train_and_score_vae () {
  local cfg_name="$1"
  local z_dim="$2"
  local bayesian="$3"
  local param_file="${PARAM_DIR}/${cfg_name}.json"
  local model_suffix="hw_${cfg_name}"

  create_param_file "${z_dim}" "${bayesian}" "${param_file}"

  if [[ ! -f "${VAE_CHECKPOINT_LOCATION}/P53_HUMAN_${model_suffix}_final" ]]; then
    python train_VAE.py \
      --MSA_data_folder "${MSA_DATA_FOLDER}" \
      --MSA_list "${MSA_LIST}" \
      --protein_index "${PROTEIN_INDEX}" \
      --MSA_weights_location "${MSA_WEIGHTS_LOCATION}" \
      --VAE_checkpoint_location "${VAE_CHECKPOINT_LOCATION}" \
      --model_name_suffix "${model_suffix}" \
      --model_parameters_location "${param_file}" \
      --training_logs_location "${TRAINING_LOGS_LOCATION}" \
      --seed "${SEED}"
  else
    echo "Checkpoint exists, skip training: ${cfg_name}"
  fi

  python compute_evol_indices.py \
    --MSA_data_folder "${MSA_DATA_FOLDER}" \
    --MSA_list "${MSA_LIST}" \
    --protein_index "${PROTEIN_INDEX}" \
    --MSA_weights_location "${MSA_WEIGHTS_LOCATION}" \
    --VAE_checkpoint_location "${VAE_CHECKPOINT_LOCATION}" \
    --model_name_suffix "${model_suffix}" \
    --model_parameters_location "${param_file}" \
    --computation_mode "all_singles" \
    --all_singles_mutations_folder "${ALL_SINGLES_MUTATIONS_FOLDER}" \
    --output_evol_indices_location "${OUTPUT_EVOL_INDICES_LOCATION}" \
    --output_evol_indices_filename_suffix "_${cfg_name}" \
    --num_samples_compute_evol_indices "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}"
}

# 1) Baseline EVE
train_and_score_vae "bayes_z50" 50 True

# 2) Latent dimension ablations
train_and_score_vae "bayes_z10" 10 True
train_and_score_vae "bayes_z100" 100 True

# 5) Bayesian vs non-Bayesian decoder
train_and_score_vae "nonbayes_z50" 50 False

# 4) Direct MSA input baseline (no VAE)
python compute_evol_indices.py \
  --MSA_data_folder "${MSA_DATA_FOLDER}" \
  --MSA_list "${MSA_LIST}" \
  --protein_index "${PROTEIN_INDEX}" \
  --MSA_weights_location "${MSA_WEIGHTS_LOCATION}" \
  --VAE_checkpoint_location "${VAE_CHECKPOINT_LOCATION}" \
  --model_name_suffix "hw_bayes_z50" \
  --model_parameters_location "${PARAM_DIR}/bayes_z50.json" \
  --computation_mode "msa_direct" \
  --all_singles_mutations_folder "${ALL_SINGLES_MUTATIONS_FOLDER}" \
  --output_evol_indices_location "${OUTPUT_EVOL_INDICES_LOCATION}" \
  --output_evol_indices_filename_suffix "_msa_direct" \
  --num_samples_compute_evol_indices 1 \
  --batch_size "${BATCH_SIZE}" \
  --msa_pseudocount 1.0

# Optional full EVE scoring pipeline for baseline run
python train_GMM_and_compute_EVE_scores.py \
  --input_evol_indices_location "./results/evol_indices" \
  --input_evol_indices_filename_suffix "_${NUM_SAMPLES}_samples_bayes_z50" \
  --protein_list "${MSA_LIST}" \
  --output_eve_scores_location "${OUTPUT_EVE_SCORES_LOCATION}" \
  --output_eve_scores_filename_suffix "hw_bayes_z50" \
  --load_GMM_models \
  --GMM_parameter_location "./results/GMM_parameters/Default_GMM_parameters" \
  --GMM_parameter_filename_suffix "default" \
  --compute_EVE_scores \
  --protein_GMM_weight 0.3 \
  --plot_histograms \
  --plot_scores_vs_labels \
  --plot_location "${PLOT_LOCATION}" \
  --labels_file_location "${LABELS_FILE}" \
  --default_uncertainty_threshold_file_location "./utils/default_uncertainty_threshold.json" \
  --verbose

python homework/analyze_results.py \
  --mapping_file "${MSA_LIST}" \
  --labels_file "${LABELS_FILE}" \
  --evol_indices_dir "${OUTPUT_EVOL_INDICES_LOCATION}" \
  --num_samples "${NUM_SAMPLES}" \
  --output_dir "${HOMEWORK_ROOT}"

echo "Homework experiments completed. See ${HOMEWORK_ROOT}."
