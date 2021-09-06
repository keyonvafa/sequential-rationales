"""Plot example of greedy rationalization for GPT-2."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import colors
from rationalization import rationalize_lm
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", 
                    type=str,
                    help="Directory where trained checkpoint is stored")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
model = AutoModelForCausalLM.from_pretrained(
  os.path.join(args.checkpoint_dir, "compatible_gpt2/checkpoint-45000"))
model.cuda()
model.eval()

input_string = ("The Supreme Court on Tuesday rejected a challenge to the "
                "constitutionality of the death penalty")
input_ids = tokenizer(input_string, return_tensors='pt')['input_ids']
# tokenizer.decode(input_ids[0])  ## Make sure string is tokenized correctly.

rationales, rationalization_log = rationalize_lm(model, 
                                                 input_ids[0].to(model.device), 
                                                 tokenizer, 
                                                 verbose=True)

input_text = rationalization_log['input_text'].copy()
rationalization_matrix = np.zeros((len(input_text), len(input_text)))
for target_ind in range(len(rationales)):
  rationalization_matrix[target_ind + 1, rationales[target_ind]] = 1

# Combine wordpieces (hackily).
# First, get a list of all the tokens that belong to a word-piece.
pieces = [i for i, x in enumerate(input_text) if x[0] != ' ' and i != 0]
for i in list(pieces):
  if i - 1 not in pieces:
    pieces.append(i - 1)
pieces = sorted(pieces)

# Next, make a list of the clusters (think of them like connected components).
clusters = []
for i in pieces:
  if (i - 1) not in [item for sublist in clusters for item in sublist]:
    new_cluster = [i]
    j = i + 1
    while j in pieces:
      new_cluster.append(j)
      j = j + 1
    clusters.append(new_cluster)

input_text = np.array(input_text)
input_text_copy = [x for x in input_text]
# Combine the rows and columns for each cluster in the wordpiece.
for cluster in clusters:
  rationalization_matrix[:, cluster[0]] = np.max(
    rationalization_matrix[:, cluster], 1)
  joined_cluster = "".join(input_text[cluster])
  input_text_copy[cluster[0]] = joined_cluster
for cluster in clusters:
  rationalization_matrix[cluster[0], :] = np.max(
    rationalization_matrix[np.array(cluster), :], 0)[np.newaxis]
input_text_copy = np.array(input_text_copy)
# Delete redundant rows and columns, moving backwards.
for cluster in clusters[::-1]:  
  for cluster_ind in cluster[1:][::-1]:
    rationalization_matrix = np.delete(rationalization_matrix, cluster_ind, 1)
    input_text_copy = np.delete(input_text_copy, cluster_ind, 0)
for cluster in clusters[::-1]:
  for cluster_ind in cluster[1:][::-1]:
    rationalization_matrix = np.delete(rationalization_matrix, cluster_ind, 0)
input_text = input_text_copy
# Zero out diagonal that results from combining wordpieces.
rationalization_matrix = np.tril(rationalization_matrix, -1)

# Make a color map of fixed colors
cmap = colors.ListedColormap(
  ['#eaeaf2', [0.01060815, 0.01808215, 0.10018654, 1.], 'firebrick'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# Highlight a row from the matrix
highlight_index = 10
rationalization_matrix[highlight_index] *= 2

# Plot.
sns.set(rc={"grid.linewidth": 8})
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(rationalization_matrix, cmap=cmap, norm=norm, 
          interpolation='nearest')
ax.set_xticks(np.arange(-0.5, len(input_text) - 0.5, 1))
ax.set_yticks(np.arange(-0.5, len(input_text) - 0.5, 1))
ax.set_xticks(range(len(input_text)), minor=True)
ax.set_yticks(range(len(input_text)), minor=True)
ax.set_xticklabels(['\n' + x for x in input_text], size=35, rotation=90, 
                   fontname="Liberation Sans")
ax.set_yticklabels(['\n' + x for x in input_text], size=35, 
                   fontname="Liberation Sans")
ax.yaxis.get_ticklabels()[highlight_index].set_color('firebrick')
ax.grid(which='minor', color='w', linestyle='-', linewidth=0)

# Save.
project_dir = os.path.abspath(
  os.path.join(os.path.dirname(__file__), os.pardir))
fig_dir = os.path.join(project_dir, "analysis/figs")
plt.savefig(
  os.path.join(fig_dir, "gpt2_rationale.pdf"), dpi=300, bbox_inches='tight')
