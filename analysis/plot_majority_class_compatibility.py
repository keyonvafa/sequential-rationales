"""Plot compatibility figures for majority class language."""
import argparse
import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fairseq import utils
from fairseq.models.transformer import TransformerModel
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import binom

sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", 
                    type=str,
                    help="Directory where trained checkpoint is stored")
args = parser.parse_args()

project_dir = os.path.abspath(
  os.path.join(os.path.dirname(__file__), os.pardir))
fairseq_dir = os.path.join(project_dir, "fairseq")
majority_class_data_dir = os.path.join(fairseq_dir, "data-bin/majority_class")

plt.rcParams["font.family"] = "Liberation Sans"

plot_titles = ["Standard training", "Compatible training"]
model_names = ["standard_majority_class", "compatible_majority_class"]

all_true_probs = []
all_model_probs = []

for model_name in model_names:
  rs = np.random.RandomState(0)
  # Load model.
  model = TransformerModel.from_pretrained(
    os.path.join(args.checkpoint_dir, model_name),
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path=majority_class_data_dir)
  if torch.cuda.is_available():
    model.cuda()
    model.half()
  model.eval()
  model.model = model.models[0]
  
  # Load data iterator.
  model.task.load_dataset('test')
  itr = model.task.get_batch_iterator(
    dataset=model.task.dataset('test'),
    max_tokens=1200,
    max_sentences=1,).next_epoch_itr(shuffle=False)

  # Only plot 100 points.
  num_evaluated = 0
  max_eval = 100

  model_probs = []
  true_probs = []

  num_evaluated = 0
  for eval_index, sample in enumerate(itr):
    num_evaluated += 1
    if torch.cuda.is_available():
      sample = utils.move_to_cuda(sample)
    target_sequence = sample['net_input']['src_tokens']
    if (sample['net_input']['src_tokens'][0, 0].item() != 
       model.task.source_dictionary.eos_index):
      # Add <eos> token to beginning of target tokens.
      target_sequence = torch.cat([
        torch.tensor([[model.task.target_dictionary.eos_index]]).to(
          target_sequence), 
        target_sequence], -1)

    sequence_string = model.decode(target_sequence[0])
    
    # Give a random input, and see how far the prediction is from the truth.
    unmasked_bits = rs.binomial(1, 0.5, 17)
    # Add 1 to unmasked bits to account for first <eos> token.
    random_indices = [[0] + list(1 + np.where(unmasked_bits == 1)[0]) + [18]]
    random_input = target_sequence[0, random_indices]
    # Fairseq adds 2 to the zero-indexed positions/
    random_positions = torch.tensor(random_indices).to(random_input) + 2
    # Evaluate transformer on partial input.
    random_decoder_out = model.model.decoder.forward(
      random_input, position_ids=random_positions)
    random_probs = model.model.get_normalized_probs(
      random_decoder_out, log_probs=False)
    one_index = model.encode("1")[0].item()
    # Find model probability that final token is 1.
    random_prob_1_majority = random_probs[0, -1, one_index].item()
    
    # Calculate true probability that final token is 1, omitting <bos> and =
    unmasked_string = np.array(
      sequence_string.split(" "))[np.where(unmasked_bits == 1)[0]]
    unmasked_1 = np.sum(unmasked_string == '1')
    unmasked_0 = np.sum(unmasked_string == '0')
    num_unmasked = unmasked_1 + unmasked_0
    num_masked = 17 - num_unmasked
    # 1 can only be successful if (9 - unmasked_1) of the num_masked are 1
    # We subtract 1 because CDF is <= rather than <
    true_prob_1_majority = 1 - binom.cdf(9 - unmasked_1 - 1, num_masked, 0.5)

    model_probs.append(random_prob_1_majority)
    true_probs.append(true_prob_1_majority)
    
    if eval_index == max_eval:
      break
  all_true_probs.append(true_probs)
  all_model_probs.append(model_probs)

# Plot points for each model.
fig = plt.figure(figsize=(6, 2))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.set_ylabel("True probability", size=14)
ax1.set_xlabel("Model probability", size=14)
ax1.set_title(plot_titles[0], size=16)
ax1.scatter(all_model_probs[0], all_true_probs[0])
ax1.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), 
         color='red', linestyle='dashed')

ax2.set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'], color='white')
ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax2.set_title(plot_titles[1], size=16)
ax2.set_xlabel("Model probability", size=14)
ax2.scatter(all_model_probs[1], all_true_probs[1])
ax2.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), 
         color='red', linestyle='dashed')

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.subplots_adjust(left=0.1, bottom=0.0, right=0.9, top=0.78, 
                    wspace=0.2, hspace=0.6)
plt.savefig("figs/majority_class_compatibility.pdf", 
            dpi=300, bbox_inches='tight')
