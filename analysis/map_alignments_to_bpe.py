"""Create mapping from alignemnts (which are not BPE) to the respective BPE."""
import argparse
import os 

import numpy as np

from fairseq.models.transformer import TransformerModel

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", 
                    type=str,
                    help="Directory where trained checkpoint is stored")
args = parser.parse_args()

project_dir = os.path.abspath(
  os.path.join(os.path.dirname(__file__), os.pardir))
fairseq_dir = os.path.join(project_dir, "fairseq")
data_path = os.path.join(
  fairseq_dir, 'data-bin/iwslt14_alignments.tokenized.de-en')
model = TransformerModel.from_pretrained(
  os.path.join(args.checkpoint_dir, 'compatible_iwslt'),
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path=data_path)

tgt_dict = model.task.target_dictionary
src_dict = model.task.source_dictionary
model.task.load_dataset('test')

itr = model.task.get_batch_iterator(
  dataset=model.task.dataset('test'),
  max_tokens=1200,
  max_sentences=1,).next_epoch_itr(shuffle=False)
num_examples = itr.total

non_bpe_srcs = [[] for _ in range(num_examples)]
non_bpe_tgts = [[] for _ in range(num_examples)]

src_bpe_maps = [[] for _ in range(num_examples)]
tgt_bpe_maps = [[] for _ in range(num_examples)]

for eval_index, sample in enumerate(itr):
  bpe_src = src_dict.string(sample['net_input']['src_tokens'], None)
  non_bpe_src = src_dict.string(
    sample['net_input']['src_tokens'], 'subword_nmt')
  bpe_tgt = tgt_dict.string(sample['net_input']['prev_output_tokens'], None)
  non_bpe_tgt = tgt_dict.string(
    sample['net_input']['prev_output_tokens'], 'subword_nmt')
  split_bpe_src = bpe_src.split(' ')
  split_bpe_tgt = bpe_tgt.split(' ')
  
  # Go through each subword in the target and find its index among full words.
  tgt_bpe_to_non_bpe_location = [0 for _ in range(len(split_bpe_tgt))]
  for i in range(1, len(split_bpe_tgt)):
    # If '@@' is present in a token, it is BPE'd, so the corresponding index
    # is equal to that of the previous wordpiece.
    if '@@' in split_bpe_tgt[i - 1]:
      tgt_bpe_to_non_bpe_location[i] = tgt_bpe_to_non_bpe_location[i - 1]
    else:
      tgt_bpe_to_non_bpe_location[i] = tgt_bpe_to_non_bpe_location[i - 1] + 1
  assert max(tgt_bpe_to_non_bpe_location) + 1 == len(non_bpe_tgt.split(" "))
  
  # Do the same for the source.
  src_bpe_to_non_bpe_location = [0 for _ in range(len(split_bpe_src))]
  for i in range(1, len(split_bpe_src)):
    if '@@' in split_bpe_src[i - 1]:
      src_bpe_to_non_bpe_location[i] = src_bpe_to_non_bpe_location[i - 1]
    else:
      src_bpe_to_non_bpe_location[i] = src_bpe_to_non_bpe_location[i - 1] + 1
  assert max(src_bpe_to_non_bpe_location) + 1 == len(non_bpe_src.split(" "))
  
  non_bpe_srcs[sample['id'].item()] = non_bpe_src
  non_bpe_tgts[sample['id'].item()] = non_bpe_tgt
  src_bpe_maps[sample['id'].item()] = src_bpe_to_non_bpe_location
  tgt_bpe_maps[sample['id'].item()] = tgt_bpe_to_non_bpe_location
  
original_data_dir = os.path.join(
  fairseq_dir, "examples/translation/iwslt14.tokenized.de-en/gold_labels")

# Make sure the number of words is the same in the original and new files.
original_src_file = os.path.join(original_data_dir, "de")
with open(original_src_file, 'r') as handle:
  original_src = [line[:-1] for line in handle]
# Since we're stripping away new lines.
original_src[-1] += "."

original_tgt_file = os.path.join(original_data_dir, "en")
with open(original_tgt_file, 'r') as handle:
  original_tgt = [line[:-1] for line in handle]
original_tgt[-1] += "."

# Make sure the files have an equal numper of examples.
assert len(original_tgt) == len(non_bpe_tgts)
assert len(original_src) == len(non_bpe_srcs)

# Make sure the length of each sequence is the same.
original_tgt_lengths = np.array([len(x.split(" ")) for x in original_tgt])
non_bp_tgt_lengths = np.array([len(x.split(" ")) for x in non_bpe_tgts])
assert np.all(original_tgt_lengths == non_bp_tgt_lengths)

original_src_lengths = np.array([len(x.split(" ")) for x in original_src])
non_bp_src_lengths = np.array([len(x.split(" ")) for x in non_bpe_srcs])
assert np.all(original_src_lengths == non_bp_src_lengths)

# Save.
with open(os.path.join(original_data_dir, 'bpe_map.de'), 'w') as f:
  for item in src_bpe_maps:
    f.write("%s\n" % ",".join([str(x) for x in item]))

with open(os.path.join(original_data_dir, 'bpe_map.en'), 'w') as f:
  for item in tgt_bpe_maps:
    f.write("%s\n" % ",".join([str(x) for x in item]))

