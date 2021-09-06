"""Evaluate rationales for alignment experiment.

This script assumes that rationales for both the greedy method and the 
baseline exist in their corresponding folders.
"""
import argparse
import json
import os
import re

import numpy as np

from collections import defaultdict
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", 
                    type=str,
                    default="gradient_norm",
                    help="Rationalization method to compare against. Must be "
                         "one of one of: 'gradient_norm', 'signed_gradient', "
                         "'integrated_gradient', 'last_attention', or "
                         "'all_attention'.")
parser.add_argument("--top_1", 
                    action='store_true',
                    help="Whether to only evaluate rationalizations on single "
                         "source word.")
args = parser.parse_args()

project_dir = os.path.abspath(
  os.path.join(os.path.dirname(__file__), os.pardir))
fairseq_dir = os.path.join(project_dir, "fairseq")

task_name = "alignments_top_1" if args.top_1 else "alignments"
greedy_dir = os.path.join(
  fairseq_dir, 
  "rationalization_results/{}/greedy".format(task_name))
baseline_dir = os.path.join(
  fairseq_dir, 
  "rationalization_results/{}/{}".format(task_name, args.baseline))

original_data_dir = os.path.join(
  fairseq_dir, "examples/translation/iwslt14.tokenized.de-en/gold_labels")
alignment_file = os.path.join(original_data_dir, "alignmentDeEn")

with open(alignment_file) as f:
  alignments = f.readlines()

# The last line is a new row, so only go up to [:-2]
alignments = [line.rstrip('\n') for line in alignments[:-2]]
split_alignments = (" ".join(alignments)).split("SENT")
alignment_sets = []

# For each sequence, store the alignment in `alignment_dict`.
# alignment_sets[example_id][word_index] contains a list of the source 
# alignments for the word at `word_index` in target `example_id`.
for index in range(1, len(split_alignments)):
  # Use both sure and possible alignments.
  split_alignment = re.split("S|P", split_alignments[index])
  alignment_dict = defaultdict(list)
  for alignment in range(1, len(split_alignment)):
    # Store the tokens referenced for each alignment.
    alignment_dict[int(split_alignment[alignment].split(" ")[2])].append(
      int(split_alignment[alignment].split(" ")[1]))
  alignment_sets.append(alignment_dict)

# Do the same as job, except using separate sure and possible alignment
# dicts so we can calculate AER.
sure_alignment_sets = []
possible_alignment_sets = []
for index in range(1, len(split_alignments)):
  split_alignment = split_alignments[index].strip().split(" ")[2:]
  sure_alignment_dict = defaultdict(list)
  possible_alignment_dict = defaultdict(list)
  for alignment in range(0, len(split_alignment), 3):
    if split_alignment[alignment] == 'S':
      sure_alignment_dict[int(split_alignment[alignment + 2])].append(
        int(split_alignment[alignment + 1]))
    elif split_alignment[alignment] == 'P':
      possible_alignment_dict[int(split_alignment[alignment + 2])].append(
        int(split_alignment[alignment + 1]))
    else:
      raise ValueError("Unrecognized link type")
  sure_alignment_sets.append(sure_alignment_dict)
  possible_alignment_sets.append(possible_alignment_dict)

src_bpe_map_file = os.path.join(original_data_dir, "bpe_map.de")
with open(src_bpe_map_file) as f:
  src_bpe_map = f.readlines()

tgt_bpe_map_file = os.path.join(original_data_dir, "bpe_map.en")
with open(tgt_bpe_map_file) as f:
  tgt_bpe_map = f.readlines()

src_bpe_map = [line.rstrip('\n') for line in src_bpe_map]
src_bpe_map = [[int(x) for x in bpe_map.split(",")] for bpe_map in src_bpe_map]

tgt_bpe_map = [line.rstrip('\n') for line in tgt_bpe_map]
tgt_bpe_map = [[int(x) for x in bpe_map.split(",")] for bpe_map in tgt_bpe_map]

# src_bpe_map[example_id][bpe_index] contains the corresponding word in the
# non-BPE'd dataset for the BPE'd source word at `bpe_index` in example 
# `example_id`. tgt_bpe_map is analogous. 

greedy_ious = []
baseline_ious = []
greedy_aers = []
baseline_aers = []
all_gold_alignment_indicators = []
all_greedy_alignment_preds = []
all_baseline_alignment_preds = []
greedy_lengths = []
baseline_lengths = []

if args.top_1:
  greedy_acc_1 = []
  baseline_acc_1 = []

for index in range(len(tgt_bpe_map)):
  greedy_file = os.path.join(greedy_dir, "{}.json".format(index))
  baseline_file = os.path.join(baseline_dir, "{}.json".format(index))
  if os.path.exists(greedy_file) and os.path.exists(baseline_file):
    with open(greedy_file) as f:
      greedy_rationalization = json.load(f)
    with open(baseline_file) as f:
      baseline_rationalization = json.load(f)
    src_len = len(greedy_rationalization['source_tokens'])
    sources_in_greedy_rationale = greedy_rationalization[
      'all_source_rationales']
    sources_in_baseline_rationale = baseline_rationalization[
      'all_source_rationales']
    for full_word_target_ind in alignment_sets[index].keys():
      # Get the set of subwords that map to `full_word_target_word_ind`
      relevant_subwords = np.where(
        np.array(tgt_bpe_map[index]) == full_word_target_ind)[0]
      # greedy_rationale will contain the union of source words for all the
      # greedy target sub-words. baseline_rationale is analogous.
      greedy_rationale = set({})
      baseline_rationale = set({})
      true_alignment = set(alignment_sets[index][full_word_target_ind])
      # Iterate through each subword that map to the same full target word.
      for relevant_subword in relevant_subwords:
        source_subwords_in_greedy_rationale = sources_in_greedy_rationale[
          relevant_subword]
        source_subwords_in_baseline_rationale = sources_in_baseline_rationale[
          relevant_subword]
        if args.top_1:
          # This is a check to make sure we're actually rationalizing 
          # the top-1 predictions.
          assert len(source_subwords_in_greedy_rationale) == 2
          assert len(source_subwords_in_baseline_rationale) == 2
          # If the first added source word is <eos>, add the second added 
          # source word.
          if source_subwords_in_greedy_rationale[0] != src_len - 1:
            greedy_rationale = greedy_rationale.union(
              [src_bpe_map[index][source_subwords_in_greedy_rationale[0]]])
          else:
            greedy_rationale = greedy_rationale.union(
              [src_bpe_map[index][source_subwords_in_greedy_rationale[1]]])
          if source_subwords_in_baseline_rationale[0] != src_len - 1:
            baseline_rationale = baseline_rationale.union(
              [src_bpe_map[index][source_subwords_in_baseline_rationale[0]]])
          else:
            baseline_rationale = baseline_rationale.union(
              [src_bpe_map[index][source_subwords_in_baseline_rationale[1]]])
          # It's accurate if all aligned words are part of the true rationale
          if len(true_alignment.intersection(greedy_rationale)) > 0:
            greedy_acc_1.append(1)
          else:
            greedy_acc_1.append(0)
          if len(true_alignment.intersection(baseline_rationale)) > 0:
            baseline_acc_1.append(1)
          else:
            baseline_acc_1.append(0)
        else:
          # Collect all the full words that contain a subword in the greedy 
          # rationale (with the exception of the <eos> token, which is always
          # at position `src_len - 1`).
          source_words_in_greedy_rationale = [
            src_bpe_map[index][x] for x in source_subwords_in_greedy_rationale 
            if x != src_len - 1]
          # We take the union of the current greedy rationale for the word
          # and the rationale for the subword we just checked.
          greedy_rationale = greedy_rationale.union(
            source_words_in_greedy_rationale)
          # Repeat for baseline rationale.
          source_words_in_baseline_rationale = [
            src_bpe_map[index][x] 
            for x in source_subwords_in_baseline_rationale 
            if x != src_len - 1]
          baseline_rationale = baseline_rationale.union(
            source_words_in_baseline_rationale)
      # Calculate metrics
      greedy_iou = len(
        true_alignment.intersection(greedy_rationale)) / len(
          true_alignment.union(greedy_rationale))
      baseline_iou = len(
        true_alignment.intersection(baseline_rationale)) / len(
          true_alignment.union(baseline_rationale))
      greedy_ious.append(greedy_iou)
      baseline_ious.append(baseline_iou)
      # Get sure and possible alignments for AER
      sure_alignment = set(sure_alignment_sets[index][full_word_target_ind])
      possible_alignment = set(possible_alignment_sets[index][
        full_word_target_ind])
      greedy_aer = 1. - (
        len(possible_alignment.intersection(greedy_rationale)) + 
        len(sure_alignment.intersection(greedy_rationale))) / (
          len(sure_alignment) + len(greedy_rationale) + 1e-6)
      baseline_aer = 1. - (
        len(possible_alignment.intersection(baseline_rationale)) + 
        len(sure_alignment.intersection(baseline_rationale))) / (
          len(sure_alignment) + len(baseline_rationale) + 1e-6)
      greedy_aers.append(greedy_aer)
      baseline_aers.append(baseline_aer)
      # Add accuracy indicators to calculate F1 score.
      all_gold_alignment_indicators.extend(
        [1 if i in true_alignment else 0 for i in range(src_len - 1)]) 
      all_greedy_alignment_preds.extend(
        [1 if i in greedy_rationale else 0 for i in range(src_len - 1)]) 
      all_baseline_alignment_preds.extend(
        [1 if i in baseline_rationale else 0 for i in range(src_len - 1)]) 
      greedy_lengths.append(len(greedy_rationale))
      baseline_lengths.append(len(baseline_rationale))

if args.top_1:
  print("Mean Acc-1: Greedy: {:.3f}, Baseline ({}): {:.3f}".format(
    np.mean(greedy_acc_1), args.baseline, np.mean(baseline_acc_1)))
else:
  greedy_f1 = f1_score(
    all_gold_alignment_indicators, all_greedy_alignment_preds)
  baseline_f1 = f1_score(
    all_gold_alignment_indicators, all_baseline_alignment_preds)
  print("Mean Length: Greedy: {:.1f}, Baseline ({}): {:.1f}".format(
    np.mean(greedy_lengths), args.baseline, np.mean(baseline_lengths)))
  print("Mean Alignment Error Rate: Greedy: {:.3f}, Baseline ({}): {:.3f}".format(
    np.mean(greedy_aers), args.baseline, np.mean(baseline_aers)))
  print("Mean IOU: Greedy: {:.3f}, Baseline ({}): {:.3f}".format(
    np.mean(greedy_ious), args.baseline, np.mean(baseline_ious)))
  print("F1 Score: Greedy: {:.3f}, Baseline ({}): {:.3f}".format(
    greedy_f1, args.baseline, baseline_f1))
