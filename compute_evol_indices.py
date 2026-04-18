import os,sys
import json
import argparse
import pandas as pd
import torch
import numpy as np

from EVE import VAE_model
from utils import data_utils


def compute_msa_direct_indices(msa_data, list_mutations_location, pseudocount=1.0, eps=1e-12):
    """
    Compute mutation scores directly from weighted MSA amino-acid frequencies.
    A higher score means the mutant residue is less supported than the WT residue.
    """
    list_mutations = pd.read_csv(list_mutations_location, header=0)

    weighted_counts = (msa_data.one_hot_encoding * msa_data.weights[:, None, None]).sum(axis=0) + pseudocount
    weighted_freqs = weighted_counts / weighted_counts.sum(axis=1, keepdims=True)

    list_valid_mutations = ['wt']
    msa_indices = [0.0]

    for mutation in list_mutations['mutations']:
        individual_substitutions = mutation.split(':')
        fully_valid_mutation = True
        score = 0.0

        for mut in individual_substitutions:
            wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
            if pos not in msa_data.uniprot_focus_col_to_wt_aa_dict or msa_data.uniprot_focus_col_to_wt_aa_dict[pos] != wt_aa or mut not in msa_data.mutant_to_letter_pos_idx_focus_list:
                print("Not a valid mutant: " + mutation)
                fully_valid_mutation = False
                break

            _, _, idx_focus = msa_data.mutant_to_letter_pos_idx_focus_list[mut]
            wt_idx = msa_data.aa_dict[wt_aa]
            mut_idx = msa_data.aa_dict[mut_aa]
            wt_freq = weighted_freqs[idx_focus, wt_idx]
            mut_freq = weighted_freqs[idx_focus, mut_idx]
            score += np.log((wt_freq + eps) / (mut_freq + eps))

        if fully_valid_mutation:
            list_valid_mutations.append(mutation)
            msa_indices.append(score)

    return list_valid_mutations, np.array(msa_indices)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Evol indices')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name is the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--computation_mode', type=str, help='Computes evol indices for all single AA mutations or for a passed in list of mutations (singles or multiples) [all_singles,input_mutations_list,msa_direct]')
    parser.add_argument('--all_singles_mutations_folder', type=str, help='Location for the list of generated single AA mutations')
    parser.add_argument('--mutations_location', type=str, help='Location of all mutations to compute the evol indices for')
    parser.add_argument('--output_evol_indices_location', type=str, help='Output location of computed evol indices')
    parser.add_argument('--output_evol_indices_filename_suffix', default='', type=str, help='(Optional) Suffix to be added to output filename')
    parser.add_argument('--num_samples_compute_evol_indices', type=int, help='Num of samples to approximate delta elbo when computing evol indices')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size when computing evol indices')
    parser.add_argument('--msa_pseudocount', default=1.0, type=float, help='Pseudocount used for direct MSA scoring in msa_direct mode')
    args = parser.parse_args()

    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: "+str(protein_name))
    print("MSA file: "+str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: "+str(theta))

    data = data_utils.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=True,
            weights_location=args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    )
    
    if args.computation_mode in ["all_singles", "msa_direct"]:
        data.save_all_singles(output_filename=args.all_singles_mutations_folder + os.sep + protein_name + "_all_singles.csv")
        args.mutations_location = args.all_singles_mutations_folder + os.sep + protein_name + "_all_singles.csv"
    else:
        args.mutations_location = args.mutations_location + os.sep + protein_name + ".csv"
        
    if args.computation_mode == "msa_direct":
        print("Computation mode: msa_direct (no VAE checkpoint needed)")
        list_valid_mutations, evol_indices = compute_msa_direct_indices(
            msa_data=data,
            list_mutations_location=args.mutations_location,
            pseudocount=args.msa_pseudocount
        )
    else:
        model_name = protein_name + "_" + args.model_name_suffix
        print("Model name: "+str(model_name))

        model_params = json.load(open(args.model_parameters_location))

        model = VAE_model.VAE_model(
                        model_name=model_name,
                        data=data,
                        encoder_parameters=model_params["encoder_parameters"],
                        decoder_parameters=model_params["decoder_parameters"],
                        random_seed=42
        )
        model = model.to(model.device)

        try:
            checkpoint_name = str(args.VAE_checkpoint_location) + os.sep + model_name + "_final"
            checkpoint = torch.load(checkpoint_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Initialized VAE with checkpoint '{}' ".format(checkpoint_name))
        except:
            print("Unable to locate VAE model checkpoint")
            sys.exit(0)
        
        list_valid_mutations, evol_indices, _, _ = model.compute_evol_indices(msa_data=data,
                                                        list_mutations_location=args.mutations_location, 
                                                        num_samples=args.num_samples_compute_evol_indices,
                                                        batch_size=args.batch_size)

    df = {}
    df['protein_name'] = protein_name
    df['mutations'] = list_valid_mutations
    df['evol_indices'] = evol_indices
    df = pd.DataFrame(df)
    
    evol_indices_output_filename = args.output_evol_indices_location+os.sep+protein_name+'_'+str(args.num_samples_compute_evol_indices)+'_samples'+args.output_evol_indices_filename_suffix+'.csv'
    try:
        keep_header = os.stat(evol_indices_output_filename).st_size == 0
    except:
        keep_header=True 
    df.to_csv(path_or_buf=evol_indices_output_filename, index=False, mode='a', header=keep_header)