"""Plot a greedy rationalization for IWSLT and compare to gold alignment."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(rc={"grid.linewidth": 8})
plt.rcParams["font.family"] = "Liberation Sans"

project_dir = os.path.abspath(
  os.path.join(os.path.dirname(__file__), os.pardir))
fairseq_dir = os.path.join(project_dir, "fairseq")
greedy_dir = os.path.join(
  fairseq_dir, 
  "rationalization_results/alignments/greedy")

greedy_file = os.path.join(greedy_dir, "257.json")
with open(greedy_file) as f:
  rationalization = json.load(f)
source_tokens = rationalization['source_tokens_text']
target_tokens = rationalization['target_tokens_text']
source_tokens[-1] = "<eos>"
target_tokens[0] = "<bos>"

# Create a matrix representing the source rationale.
rationale_matrix = np.zeros(
  (len(target_tokens) - 2, len(source_tokens))) + float("inf")
for target_ind in range(0, len(rationalization['all_source_rationales']) - 1):
  rationale_matrix[
    target_ind, rationalization['all_source_rationales'][target_ind]] = 1

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(1, 1, 1)

# Combine wordpieces for plot
source_pieces = [i for i, x in enumerate(source_tokens) if '@@' in x]
target_pieces = [i for i, x in enumerate(target_tokens) if '@@' in x]
for i in list(source_pieces):
  if i + 1 not in source_pieces:
    source_pieces.append(i + 1)
for i in list(target_pieces):
  if i in target_pieces and i + 1 not in target_pieces:
    target_pieces.append(i + 1)
source_pieces = sorted(source_pieces)
target_pieces = sorted(target_pieces)
source_clusters = []
target_clusters = []
# Hacky way to find BPE clusters, and this probably doesn't generally work,
# but it works for this specific example.
for i in source_pieces:
  if (i - 1) not in source_pieces:
    new_cluster = [i]
    j = i + 1
    while j in source_pieces:
      new_cluster.append(j)
      j = j + 1
    source_clusters.append(new_cluster)
for i in target_pieces:
  if (i - 1) not in target_pieces:
    new_cluster = [i]
    j = i + 1
    while j in target_pieces:
      new_cluster.append(j)
      j = j + 1
    target_clusters.append(new_cluster)
source_tokens = np.array(source_tokens)
target_tokens = np.array(target_tokens)
source_tokens_copy = [x for x in source_tokens] 
target_tokens_copy = [x for x in target_tokens]
# combine rows in the matrix
for cluster in source_clusters:
  rationale_matrix[:, cluster[0]] = np.min(rationale_matrix[:, cluster], 1)
  joined_cluster = "".join(
    [x[:-2] if '@@' in x else x for x in source_tokens[cluster]])
  source_tokens_copy[cluster[0]] = joined_cluster
for cluster in target_clusters:
  rationale_matrix[cluster[0] - 1, :] = np.min(
    rationale_matrix[np.array(cluster) - 1, :], 0)[np.newaxis]
  joined_cluster = "".join(
    [x[:-2] if '@@' in x else x for x in target_tokens[cluster]])
  target_tokens_copy[cluster[0]] = joined_cluster
source_tokens_copy = np.array(source_tokens_copy)
# delete rows in the matrix, going backwards to make sure indices aren't stale
for cluster in source_clusters[::-1]:  
  for cluster_ind in cluster[1:][::-1]:
    rationale_matrix = np.delete(rationale_matrix, cluster_ind, 1)
    source_tokens_copy = np.delete(source_tokens_copy, cluster_ind, 0)
target_tokens_copy = np.array(target_tokens_copy)
for cluster in target_clusters[::-1]:
  for cluster_ind in cluster[1:][::-1]:
    rationale_matrix = np.delete(rationale_matrix, cluster_ind - 1, 0)
    target_tokens_copy = np.delete(target_tokens_copy, cluster_ind, 0)
source_tokens = source_tokens_copy
target_tokens = target_tokens_copy

ax.imshow(rationale_matrix)
ax.set_xticks(np.arange(-.5, len(source_tokens) - 0.5))
ax.set_yticks(np.arange(-.5, len(target_tokens) - 2.5))
ax.set_xticks(range(len(source_tokens)), minor=True)
ax.set_yticks(range(len(target_tokens) - 2), minor=True)
ax.set_xticklabels(['\n\n' + x for x in source_tokens], size=35, rotation=90)
ax.set_yticklabels(['\n\n' + y for y in target_tokens[1:-1]], size=35)
ax.grid(which='minor', color='w', linestyle='-', linewidth=0)
plt.savefig("figs/iwslt_rationalization.pdf", dpi=300, bbox_inches='tight')

