"""Compare rationalization durations for analogies experiment.

This script assumes we've already performed rationalization for the 
analogies experiment, using `rationalize_analogies.py`. We compare three
methods: greedy rationalization (with efficient batching), greedy 
rationalization (with inefficient batching) and exhaustive rationalization.
Greedy rationalization with inefficient batching always produces the same
results as greedy rationalization, but it doesn't evaluate on sparse subsets.
Rather, it passes in masked inputs. We don't actually implement it, but we can
simulate the length by repeatedly evaluating the transformer on the set of all
inputs. 
"""
import argparse
import json
import os
import time
import torch

import numpy as np

from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", 
                    type=str,
                    help="Directory where trained checkpoint is stored")
args = parser.parse_args()


model = AutoModelForCausalLM.from_pretrained(
  os.path.join(args.checkpoint_dir, "compatible_gpt2/checkpoint-45000"))
model.cuda()
model.eval()

huggingface_dir = os.path.dirname(__file__)
greedy_dir = os.path.join(
  huggingface_dir, 
  "rationalization_results/analogies/greedy")
exhaustive_dir = os.path.join(
  huggingface_dir, 
  "rationalization_results/analogies/exhaustive")

efficient_greedy_durations = []
inefficient_greedy_durations = []
exhaustive_durations = []

for label_index in range(15):  # there are 15 analogy categories
  for analogy_index in range(100):  # 100 is more than we ever do
    greedy_file = os.path.join(greedy_dir, "{}_{}.json".format(
      label_index, analogy_index))
    exhaustive_file = os.path.join(exhaustive_dir, "{}_{}.json".format(
      label_index, analogy_index))
    if os.path.exists(greedy_file):
      with open(greedy_file) as f:
        greedy_rationalization = json.load(f)
      greedy_rationale = greedy_rationalization["all_rationales"][0]
      efficient_greedy_durations.append(greedy_rationalization['duration'])
      # Now, simulate greedy inefficient search.
      all_context_tokens = torch.tensor(
        greedy_rationalization["input_ids"][:-1])[None].to(model.device)
      inefficient_start_time = time.time()
      with torch.no_grad():
        # Run for-loop as if we're evlauating all the inputs, one evaluation per
        # rationalization size.
        for rationale_size in range(1, len(greedy_rationale)):
          if rationale_size == 1:
            _ = model(all_context_tokens)
          else:
            _ = model(all_context_tokens.repeat(
              [len(all_context_tokens[0]) - rationale_size, 1]))
        inefficient_end_time = time.time()
        inefficient_greedy_durations.append(
          inefficient_end_time - inefficient_start_time)
    if os.path.exists(exhaustive_file):
      # NOTE: the exhaustive rationale only exists for smaller rationales, so
      # the average time will be quite a bit longer than the recorded one.
      with open(exhaustive_file) as f:
        exhaustive_rationalization = json.load(f)
      exhaustive_durations.append(exhaustive_rationalization['duration'])

print("Average greedy time: {:.2f}".format(
  np.mean(efficient_greedy_durations)))
print("Average inefficient greedy time: {:.2f}".format(
  np.mean(inefficient_greedy_durations)))
print("Average exhaustive time: {:.2f}".format(
  np.mean(exhaustive_durations)))
