"""Evaluate rationales for annotated Lambada dataset.

This script assumes that rationales for both the greedy method and the 
baseline exist in their corresponding folders.
"""
import argparse
import json
import os

import numpy as np

from sklearn.metrics import f1_score

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
  "rationalization_results/lambada/greedy")
baseline_dir = os.path.join(
  huggingface_dir, 
  "rationalization_results/lambada/{}".format(args.baseline))

greedy_lengths = []
greedy_ious = []
word_in_greedy_rationale_indicators = []

baseline_lengths = []
baseline_ious = []
baseline_f1s = []
word_in_baseline_rationale_indicators = []

word_in_human_rationale_indicators = []

for label_index in range(107):  # there are 107 annotated rationales.
  greedy_file = os.path.join(greedy_dir, "{}.json".format(label_index))
  baseline_file = os.path.join(baseline_dir, "{}.json".format(label_index))
  if os.path.exists(greedy_file) and os.path.exists(baseline_file):
    with open(greedy_file) as f:
      greedy_rationalization = json.load(f)
    with open(baseline_file) as f:
      baseline_rationalization = json.load(f)
    
    human_rationale = set(greedy_rationalization['human_subword_rationale'])
    # In case the target word contains multiple subwords, we consider the
    # rationale of the target word to be the union of all the subword 
    # rationales.
    greedy_rationale = set(
      [item for sublist in greedy_rationalization['all_rationales'] 
       for item in sublist])
    baseline_rationale = set(
      [item for sublist in baseline_rationalization['all_rationales'] 
       for item in sublist])

    # If the target word contains multiple subwords, we don't penalize the
    # rationales of the later subwords for containing the target word in their
    # rationales.
    num_words_before_target = greedy_rationalization[
      'rationalization'][0]['target_position']
    greedy_rationale = set(
      [x for x in greedy_rationale if x <= num_words_before_target])
    baseline_rationale = set(
      [x for x in baseline_rationale if x <= num_words_before_target])

    greedy_lengths.append(len(greedy_rationale))
    baseline_lengths.append(len(baseline_rationale))

    greedy_intersection = greedy_rationale.intersection(human_rationale)
    greedy_union = greedy_rationale.union(human_rationale)
    greedy_ious.append(len(greedy_intersection) / len(greedy_union))
    baseline_intersection = baseline_rationale.intersection(human_rationale)
    baseline_union = baseline_rationale.union(human_rationale)
    baseline_ious.append(len(baseline_intersection) / len(baseline_union))

    word_in_greedy_rationale_indicators.extend([
      1 if i in greedy_rationale 
      else 0 for i in range(num_words_before_target)])
    word_in_baseline_rationale_indicators.extend([
      1 if i in baseline_rationale else 0 
      for i in range(num_words_before_target)])
    word_in_human_rationale_indicators.extend([
      1 if i in human_rationale else 0
      for i in range(num_words_before_target)])

greedy_f1 = f1_score(word_in_human_rationale_indicators, 
                     word_in_greedy_rationale_indicators)
baseline_f1 = f1_score(word_in_human_rationale_indicators,
                       word_in_baseline_rationale_indicators)
print("Mean Length: Greedy: {:.1f}, Baseline ({}): {:.1f}".format(
  np.mean(greedy_lengths), args.baseline, np.mean(baseline_lengths)))
print("Mean IOU: Greedy: {:.2f}, Baseline ({}): {:.2f}".format(
  np.mean(greedy_ious), args.baseline, np.mean(baseline_ious)))
print("Mean F1: Greedy: {:.2f}, Baseline ({}): {:.2f}".format(
  np.mean(greedy_f1), args.baseline, np.mean(baseline_f1)))
