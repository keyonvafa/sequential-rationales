"""Evaluate rationales for analogies experiment.

This script assumes that rationales for both the greedy method and the 
baseline exist in their corresponding folders.
"""
import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", 
                    type=str,
                    default="gradient_norm",
                    help="Rationalization method to compare against. Must be "
                         "one of one of: 'gradient_norm', 'signed_gradient', "
                         "'integrated_gradient', 'last_attention', or "
                         "'all_attention'.")
args = parser.parse_args()

project_dir = os.path.abspath(
  os.path.join(os.path.dirname(__file__), os.pardir))
huggingface_dir = os.path.join(project_dir, "huggingface")

greedy_dir = os.path.join(
  huggingface_dir, 
  "rationalization_results/analogies/greedy")
baseline_dir = os.path.join(
  huggingface_dir, 
  "rationalization_results/analogies/{}".format(args.baseline))
exhaustive_dir = os.path.join(
  huggingface_dir, 
  "rationalization_results/analogies/exhaustive")

greedy_lengths = []
greedy_no_distractors = []
greedy_approximation_ratios = []
greedy_contain_antecedents = []

baseline_lengths = []
baseline_no_distractors = []
baseline_approximation_ratios = []
baseline_contain_antecedents = []

for label_index in range(15):  # there are 15 analogy categories
  for analogy_index in range(100):  # 100 is more than we ever do
    greedy_file = os.path.join(greedy_dir, "{}_{}.json".format(
      label_index, analogy_index))
    baseline_file = os.path.join(baseline_dir, "{}_{}.json".format(
      label_index, analogy_index))
    exhaustive_file = os.path.join(exhaustive_dir, "{}_{}.json".format(
      label_index, analogy_index))
    if os.path.exists(greedy_file) and os.path.exists(baseline_file):
      with open(greedy_file) as f:
        greedy_rationalization = json.load(f)
      with open(baseline_file) as f:
        baseline_rationalization = json.load(f)
      greedy_rationale = greedy_rationalization["all_rationales"][0]
      baseline_rationale = baseline_rationalization["all_rationales"][0]
      antecedent_index = greedy_rationalization['antecedent_index']
      distractor_start = greedy_rationalization['distractor_start']
      distractor_end = greedy_rationalization['distractor_end']
      distractor_set = set(range(distractor_start, distractor_end + 1))
      greedy_lengths.append(len(greedy_rationale))
      baseline_lengths.append(len(baseline_rationale))
      greedy_no_distractors.append(
        1 if len(set(greedy_rationale).intersection(distractor_set)) == 0 
        else 0)
      baseline_no_distractors.append(
        1 if len(set(baseline_rationale).intersection(distractor_set)) == 0
        else 0)
      greedy_contain_antecedents.append(
        1 if len(
          set(greedy_rationale).intersection(set([antecedent_index]))) == 1 
        else 0)
      baseline_contain_antecedents.append(
        1 if len(
          set(baseline_rationale).intersection(set([antecedent_index]))) == 1
        else 0)
      if os.path.exists(exhaustive_file):
        with open(exhaustive_file) as f:
          exhaustive_rationalization = json.load(f)
        exhaustive_rationale = exhaustive_rationalization["all_rationales"][0]
        greedy_approximation_ratios.append(
          len(greedy_rationale) / len(exhaustive_rationale))
        baseline_approximation_ratios.append(
          len(baseline_rationale) / len(exhaustive_rationale))
   
print("Mean Length: Greedy: {:.1f}, Baseline ({}): {:.1f}".format(
  np.mean(greedy_lengths), args.baseline, np.mean(baseline_lengths)))
print("Mean Approximation Ratio: Greedy: {:.2f}, "
      "Baseline ({}): {:.2f}".format(np.mean(greedy_approximation_ratios), 
                                     args.baseline, 
                                     np.mean(baseline_approximation_ratios)))
print("Mean 'Contains Antecedent': Greedy: {:.2f}, "
      "Baseline ({}): {:.2f}".format(np.mean(greedy_contain_antecedents), 
                                     args.baseline, 
                                     np.mean(baseline_contain_antecedents)))
print("Mean 'No Distractors': Greedy: {:.2f}, "
      "Baseline ({}): {:.2f}".format(np.mean(greedy_no_distractors), 
                                     args.baseline, 
                                     np.mean(baseline_no_distractors)))
