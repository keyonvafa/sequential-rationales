"""Evaluate rationales for distractor experiment.

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
fairseq_dir = os.path.join(project_dir, "fairseq")

# Load break indices.
with open(os.path.join(fairseq_dir, 'generated_translations/breaks.de')) as f:
  breaks_de = f.readlines()

with open(os.path.join(fairseq_dir, 'generated_translations/breaks.en')) as f:
  breaks_en = f.readlines()

breaks_de = [int(line.rstrip('\n')) for line in breaks_de]
breaks_en = [int(line.rstrip('\n')) for line in breaks_en]

# The variables that end with `no_distractors` collect indicators for whenever
# the rationale contains 0 distractors. The variables that end with 
# `num_distractors` contain the number of distractors in the rationale.
greedy_source_no_distractors = []
baseline_source_no_distractors = []
greedy_source_num_distractors = []
baseline_source_num_distractors = []

greedy_target_no_distractors = []
baseline_target_no_distractors = []
greedy_target_num_distractors = []
baseline_target_num_distractors = []

for index in range(len(breaks_en)):
  # Load rationalization results.
  greedy_rationale_dir = os.path.join(
    fairseq_dir, 
    'rationalization_results/distractors/greedy')
  baseline_rationale_dir = os.path.join(
    fairseq_dir, 
    'rationalization_results/distractors/{}'.format(args.baseline))
  greedy_file = os.path.join(greedy_rationale_dir, "{}.json".format(index))
  baseline_file = os.path.join(baseline_rationale_dir, "{}.json".format(index))
  if os.path.exists(greedy_file) and os.path.exists(baseline_file):
    with open(greedy_file) as f:
      greedy_rationales = json.load(f)
    with open(baseline_file) as f:
      baseline_rationales = json.load(f)
    greedy_source_rationales = greedy_rationales['all_source_rationales']
    greedy_target_rationales = greedy_rationales['all_target_rationales']
    baseline_source_rationales = baseline_rationales['all_source_rationales']
    baseline_target_rationales = baseline_rationales['all_target_rationales']

    # For each rationale, see how many distractors are contained.
    for target_ind in range(len(greedy_source_rationales)):
      greedy_sources = np.array(greedy_source_rationales[target_ind])
      baseline_sources = np.array(baseline_source_rationales[target_ind])
      # Since the first token in each target rationale is always the
      # previous token, we don't include it in case it happens to be the 
      # cross-sequence boundary.
      greedy_targets = np.array(greedy_target_rationales[target_ind])[1:]
      baseline_targets = np.array(baseline_target_rationales[target_ind])[1:]
      if target_ind < breaks_en[index]:
        # If we are rationalizing a target from before the boundary, we can
        # only consider source-side crossovers.
        # We treat the source-side <eos> token as a special token, so we don't
        # penalize rationales for including it.
        greedy_sources = greedy_sources[
          greedy_sources != len(greedy_rationales['source_tokens']) - 1]
        baseline_sources = baseline_sources[
          baseline_sources != len(baseline_rationales['source_tokens']) - 1]
        # Penalize each source rationale that contains a source token from 
        # after the source-side boundary.
        if np.all(greedy_sources < breaks_de[index]):
          greedy_source_no_distractors.append(1)
        else:
          greedy_source_no_distractors.append(0)
        if np.all(baseline_sources < breaks_de[index]):
          baseline_source_no_distractors.append(1)
        else:
          baseline_source_no_distractors.append(0)
        greedy_source_num_distractors.append(
          np.sum(greedy_sources >= breaks_de[index]))
        baseline_source_num_distractors.append(
          np.sum(baseline_sources >= breaks_de[index]))
      else:
        # If we are rationalizing a target after the boundary, we can consider
        # both source-side and target-side crossovers.
        # First, source-side crossovers:
        if np.all(greedy_sources >= breaks_de[index]):
          greedy_source_no_distractors.append(1)
        else:
          greedy_source_no_distractors.append(0)
        if np.all(baseline_sources >= breaks_de[index]):
          baseline_source_no_distractors.append(1)
        else:
          baseline_source_no_distractors.append(0)
        # Now, for target-side crossovers, we don't penalize target rationales
        # for containing the target first token since it is special.
        greedy_targets = greedy_targets[greedy_targets != 0]
        baseline_targets = baseline_targets[baseline_targets != 0]
        if np.all(greedy_targets > breaks_en[index]):
          greedy_target_no_distractors.append(1)
        else:
          greedy_target_no_distractors.append(0)
        if np.all(baseline_targets > breaks_en[index]):
          baseline_target_no_distractors.append(1)
        else:
          baseline_target_no_distractors.append(0)
        
        greedy_source_num_distractors.append(
          np.sum(greedy_sources < breaks_de[index]))
        baseline_source_num_distractors.append(
          np.sum(baseline_sources < breaks_de[index]))
        greedy_target_num_distractors.append(
          np.sum(greedy_targets <= breaks_en[index]))
        baseline_target_num_distractors.append(
          np.sum(baseline_targets <= breaks_en[index]))


print("Fraction with source disctractors: Greedy: {:.3f}, "
      "Baseline ({}): {:.3f}".format(
        1. - np.mean(greedy_source_no_distractors), 
        args.baseline,
        1. - np.mean(baseline_source_no_distractors)))
print("Fraction with target distractors: Greedy: {:.3f}, "
      "Baseline ({}): {:.3f}".format(
        1. - np.mean(greedy_target_no_distractors), 
        args.baseline,
        1. - np.mean(baseline_target_no_distractors)))
print("......................")
print("Mean source distractors: Greedy: {:.3f}, "
      "Baseline ({}): {:.3f}".format(
        np.mean(greedy_source_num_distractors), 
        args.baseline,
        np.mean(baseline_source_num_distractors)))
print("Mean target distractors: Greedy: {:.3f}, "
      "Baseline ({}): {:.3f}".format(
        np.mean(greedy_target_num_distractors), 
        args.baseline,
        np.mean(baseline_target_num_distractors)))
