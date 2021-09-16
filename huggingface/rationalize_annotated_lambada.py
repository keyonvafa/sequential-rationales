"""Rationalize annotated LAMBADA dataset.

The original dataset is provided by [1]. We annotated the dataset to include
the rationales for the final words of a sample of the passages. For more 
details, refer to our paper.

[1] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, 
Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel 
Fernández. 2016. The LAMBADA dataset: Word prediction requiring a broad 
discourse context. In Association for Computational Linguistics.
"""
import argparse
import os
import json
import time

import numpy as np
import pandas as pd

from rationalization import baseline_rationalize_lm, rationalize_lm
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


huggingface_dir = os.path.dirname(__file__)
project_dir = os.path.abspath(os.path.join(huggingface_dir, os.pardir))

# Load annotated lambada.
df = pd.read_json(os.path.join(project_dir, 'annotated_lambada.json'), 
                  orient='records', lines=True)
# df has 3 columns:
# - lamabdaIndex: The index of the example from the original LAMBADA dataset.
# - text: The text of the example (unchanged from the original dataset).
# - rationale: The rationales for the final word of each passage.

# Load model
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
model = AutoModelForCausalLM.from_pretrained(
  os.path.join(args.checkpoint_dir, "compatible_gpt2/checkpoint-45000"))
model.cuda()
model.eval()

# Rationalize each example in dataset.
first_time = time.time()
for row_num in range(len(df)):
  start_time = time.time()
  row = df.iloc[row_num]
  text = row['text']
  rationale = row['rationale']
  # GPT-2 tokenizes the text into subwords, but the annotators in the dataset
  # used full words. So we need to create a mapping between the subwords and
  # the original words.
  tokenized_output = tokenizer(text, return_offsets_mapping=True)
  offset_mapping = tokenized_output['offset_mapping']
  subword_rationale = []
  for full_word_rationale in rationale:
    full_word_first_char = full_word_rationale[0]
    full_word_last_char = full_word_rationale[1]
    for offset_ind, offset in enumerate(offset_mapping):
      if full_word_first_char < offset[1] and full_word_last_char >= offset[1]:
        subword_rationale.append(offset_ind)
  all_ids = tokenizer(text, return_tensors='pt')['input_ids']
  # Check if the words indexed by the subword rationale...
  # [tokenizer.decode([x]) for x in np.array(all_ids[0])[subword_rationale]]
  # ...are equal to the full words in the original rationale.
  # text[word_rationale[0]:word_rationale[1]] for word_rationale in rationale]
  # Find index where the last word ends, since we only need to rationalize the
  # last word.
  last_word_removed = ' '.join(text.split(' ')[:-1])
  tokenized_last_word_removed = tokenizer(
    last_word_removed, return_tensors='pt')['input_ids']
  start_step = len(tokenized_last_word_removed[0]) - 1

  if args.method == 'greedy':
    all_rationales, rationalization_log = rationalize_lm(
      model,
      all_ids[0].cuda(),
      tokenizer,
      verbose=args.verbose,
      start_step=start_step,
    )
  else:
    all_rationales, rationalization_log = baseline_rationalize_lm(
      model,
      all_ids[0].cuda(),
      tokenizer,
      method=args.method,
      verbose=args.verbose,
      start_step=start_step,
    )
  rationalization_log['human_subword_rationale'] = subword_rationale
  # Save rationalization results
  results_dir = os.path.join(
    huggingface_dir, 
    "rationalization_results/lambada/{}".format(args.method))
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  file_name = os.path.join(
    results_dir, "{}.json".format(row_num))
  print("...writing to {}".format(file_name))
  with open(file_name, 'w') as outfile:
    json.dump(rationalization_log, outfile, default=convert)
  print("...finished in {:.2f} (average: {:.2f})".format(
    time.time() - start_time, 
    (time.time() - first_time) / (row_num + 1)))


