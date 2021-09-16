"""Perform rationalization for IWSLT experiments.

Use this script to perform rationalization for either the distractor 
experiment or for the alignment experiment.
"""
import argparse
import json
import os
import time
import torch

import numpy as np

from fairseq import utils
from fairseq.models.transformer import TransformerModel
from rationalization import baseline_rationalize_conditional_model, rationalize_conditional_model

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", 
                    type=str,
                    help="Directory where trained checkpoint is stored")
parser.add_argument("--task", 
                    type=str,
                    help="Experiment to rationalize ('alignments' or "
                         "'distractors')")
parser.add_argument("--method", 
                    type=str,
                    default="greedy",
                    help="Rationalization method. Must be one of: 'greedy', "
                         "'gradient_norm', 'signed_gradient', "
                         "'integrated_gradient', 'last_attention', or "
                         "'all_attention'.")
parser.add_argument("--verbose", 
                    action='store_true',
                    help="Whether to print rationalization results.")
parser.add_argument("--top_1", 
                    action='store_true',
                    help="Whether to only rationalize a single source word "
                         "(only used if --task is set to 'alignments'.")
parser.add_argument("--max_steps", 
                    type=int,
                    default=1024,
                    help="Maximum number of steps to perform rationalization.")
args = parser.parse_args()


def convert(o):
  """Helper function to convert dict with NumPy entries to JSON file."""
  if isinstance(o, np.int64): 
    return int(o)  
  raise TypeError


fairseq_dir = os.path.dirname(__file__)

if args.task == 'distractors':
  if args.top_1:
    raise ValueError("It doesn't make sense to perform top-1 rationalization "
                     "for the distractors experiment.")
  data_path = os.path.join(fairseq_dir,
                           'data-bin/iwslt14_distractors.tokenized.de-en')
elif args.task == 'alignments':
  data_path = os.path.join(fairseq_dir,
                           'data-bin/iwslt14_alignments.tokenized.de-en')
else:
  raise ValueError("--task must be either 'distractors' or 'alignments'.")

model = TransformerModel.from_pretrained(
  os.path.join(args.checkpoint_dir, 'compatible_iwslt'),
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path=data_path)
model.half()
model.cuda()
model.eval()

# Make iterator for the data.
model.task.load_dataset('test')
itr = model.task.get_batch_iterator(
  dataset=model.task.dataset('test'),
  max_tokens=1200,
  max_sentences=1,).next_epoch_itr(shuffle=False)

# Shortcut for model indexing.
model.model = model.models[0]

rs = np.random.RandomState(0)
indices_to_evaluate = np.sort(rs.choice(itr.total, 60, replace=False))

first_time = time.time()
for eval_index, sample in enumerate(itr):
  print("Working on {}/{}...".format(eval_index, itr.total))
  start_time = time.time()
  sample = utils.move_to_cuda(sample)
  if sample['target'][0, 0].item() != model.task.source_dictionary.eos_index:
    # Add <eos> token to beginning of target tokens.
    sample['target'] = torch.cat([
      torch.tensor([[model.task.target_dictionary.eos_index]]).to(
        sample['target']), sample['target']], -1)
  if args.method == 'greedy':
    (source_rationales, target_rationales, 
     rationalization_log) = rationalize_conditional_model(
       model,
       sample['net_input']['src_tokens'][0], 
       sample['target'][0], 
       verbose=args.verbose,
       max_steps=args.max_steps,
       top_1=args.top_1)
  else:
    (source_rationales, target_rationales, 
     rationalization_log) = baseline_rationalize_conditional_model(
       model,
       sample['net_input']['src_tokens'][0], 
       sample['target'][0],
       args.method,
       verbose=args.verbose,
       max_steps=args.max_steps,
       top_1=args.top_1)

  # Save rationalization results
  task_name = ("alignments_top_1" 
               if (args.top_1 and args.task == 'alignments') 
               else args.task)
  results_dir = os.path.join(
    fairseq_dir, 
    "rationalization_results/{}/{}".format(task_name, args.method))
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)
  file_name = os.path.join(results_dir, "{}.json".format(
    str(sample['id'].item())))
  print("...writing to {}".format(file_name))
  with open(file_name, 'w') as outfile:
    json.dump(rationalization_log, outfile, default=convert)
    print("...finished in {:.2f} (average: {:.2f})".format(
      time.time() - start_time, (time.time() - first_time) / (eval_index + 1)))

