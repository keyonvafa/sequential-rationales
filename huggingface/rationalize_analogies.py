"""Rationalize templated analogies using data from [1].

[1] Tomas Mikolov, Kai Chen, Greg S. Corrado, and Jeff Dean. 2013. Efficient
estimation of word representations in vector space. In Workshop Track at ICLR.
"""
import argparse
import os
import json
import time
import torch

import numpy as np

from data_utils import create_analogy_templates, preprocess_analogies
from rationalization import baseline_rationalize_lm, exhaustive_rationalize_lm, rationalize_lm
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", 
                    type=str,
                    help="Directory where trained checkpoint is stored")
parser.add_argument("--method", 
                    type=str,
                    default="greedy",
                    help="Rationalization method. Must be one of: 'greedy', "
                         "'gradient_norm', 'signed_gradient', "
                         "'integrated_gradient', 'attention_rollout', "
                         "'last_attention', 'all_attention', or "
                         "'exhaustive'.")
parser.add_argument("--verbose", 
                    action='store_true',
                    help="Whether to print rationalization results.")
args = parser.parse_args()


def convert(o):
  """Helper function to convert dict with NumPy entries to JSON file."""
  if isinstance(o, np.int64): 
    return int(o)  
  raise TypeError


# Create random state since we only perform exhaustive search on a random set
# of indices.
rs = np.random.RandomState(0)

tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
model = AutoModelForCausalLM.from_pretrained(
  os.path.join(args.checkpoint_dir, "compatible_gpt2/checkpoint-45000"))
model.cuda()
model.eval()

huggingface_dir = os.path.dirname(__file__)
analogies_file = os.path.join(huggingface_dir, "data/analogies.txt")
with open(analogies_file) as f:
  analogies = f.readlines()
analogies = [line.rstrip('\n') for line in analogies]

all_analogies = preprocess_analogies(analogies, tokenizer)
all_analogies = create_analogy_templates(all_analogies)

analogy_labels = list(all_analogies.keys())
num_rationalized = 0
first_time = time.time()
for label_index, analogy_label in enumerate(analogy_labels):
  template = all_analogies[analogy_label]['template']
  # Find which word is the antecedent and which is the target.
  target_kind = "a" if template.index("[A]") > template.index("[B]") else "b"
  antecedent_kind = "a" if target_kind == "b" else "b"
  for index in range(len(all_analogies[analogy_label]['a'])):
    # Since exhaustive search is very time-consuming, we only perform 
    # exhaustive search for a random third of the inputs.
    if (args.method != 'exhaustive') or rs.random() < 0.33:
      word_a = all_analogies[analogy_label]['a'][index]
      word_b = all_analogies[analogy_label]['b'][index]
      antecedent_word = all_analogies[analogy_label][antecedent_kind][index]
      target_word = all_analogies[analogy_label][target_kind][index]
      full_sentence = template.replace("[A]", word_a).replace("[B]", word_b)
      full_ids = tokenizer(full_sentence, return_tensors='pt')[
        'input_ids'].cuda()
      model_prediction = tokenizer.decode(model.generate(full_ids[:, :-1], 
                                          max_length=len(full_ids[0]), 
                                          do_sample=False)[0])
      distractor_start_id = tokenizer.encode(" (")[0]
      distractor_end_id = tokenizer.encode(").")[0]
      distractor_start = np.where(
        full_ids[0].cpu().numpy() == distractor_start_id)[0][0]
      distractor_end = np.where(
        full_ids[0].cpu().numpy() == distractor_end_id)[0][0]
      distractor_removed = torch.cat(
        [full_ids[:, :distractor_start], full_ids[:, distractor_end + 1:]], 1)
      model_prediction_no_distractor = tokenizer.decode(
        model.generate(distractor_removed[:, :-1], 
                       max_length=len(distractor_removed[0]), 
                       do_sample=False)[0])
      # Only rationalize analogies the model predicts correctly, both with
      # and without the distractor
      if (model_prediction.split(" ")[-1] == target_word and 
         model_prediction_no_distractor.split(" ")[-1] == target_word):
        print("Rationalizing analogy {}, example {}...".format(
          label_index, index))
        start_time = time.time()
        num_rationalized += 1
        antecedent_word_id = tokenizer.encode(" " + antecedent_word)[0]
        antecedent_index = np.where(
          full_ids.cpu().numpy()[0] == antecedent_word_id)[0][-1]
        if args.method == 'greedy':
          all_rationales, rationalization_log = rationalize_lm(
            model,
            full_ids[0],
            tokenizer,
            verbose=args.verbose,
            start_step=len(full_ids[0]) - 2,
          )
        elif args.method in ['gradient_norm', 'signed_gradient', 
                             'integrated_gradient', 'attention_rollout', 
                             'last_attention', 'all_attention']:
          all_rationales, rationalization_log = baseline_rationalize_lm(
            model,
            full_ids[0],
            tokenizer,
            method=args.method,
            verbose=args.verbose,
            start_step=len(full_ids[0]) - 2,
          )
        elif args.method == 'exhaustive':
          all_rationales, rationalization_log = exhaustive_rationalize_lm(
            model,
            full_ids[0],
            tokenizer,
            verbose=args.verbose,
            max_steps=6,
            start_step=len(full_ids[0]) - 2,
          )
        else:
          raise ValueError("Unrecognized method.")
        # Save, unless we're exhaustively rationalizing and exhaustive search 
        # didn't complete in a reasonable amount of time.
        if all_rationales is not None:
          # Add information we'll need to evaluate rationales.
          rationalization_log['antecedent_index'] = antecedent_index
          rationalization_log['distractor_start'] = distractor_start
          rationalization_log['distractor_end'] = distractor_end
          # Save rationalization results
          results_dir = os.path.join(
            huggingface_dir, 
            "rationalization_results/analogies/{}".format(args.method))
          if not os.path.exists(results_dir):
            os.makedirs(results_dir)
          file_name = os.path.join(
            results_dir, "{}_{}.json".format(label_index, index))
          print("...writing to {}".format(file_name))
          with open(file_name, 'w') as outfile:
            json.dump(rationalization_log, outfile, default=convert)
        print("...finished in {:.2f} (average: {:.2f})".format(
          time.time() - start_time, 
          (time.time() - first_time) / (num_rationalized + 1)))
