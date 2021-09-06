"""Plot GPT-2 probabilites for 'the' followed by 'the'."""
import argparse
import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", 
                    type=str,
                    help="Directory where trained checkpoint is stored")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
pretrained_gpt2 = AutoModelForCausalLM.from_pretrained('gpt2-medium')
compatible_gpt2 = AutoModelForCausalLM.from_pretrained(
  os.path.join(args.checkpoint_dir, "compatible_gpt2/checkpoint-45000"))

pretrained_gpt2.cuda()
pretrained_gpt2.eval()
compatible_gpt2.cuda()
compatible_gpt2.eval()

models = [pretrained_gpt2, compatible_gpt2]
model_names = ["Pretrained", "Finetuned for compatibility"]

all_the_probs = [[] for _ in range(len(models))]

for model_ind, model in enumerate(models):
  the_probs = all_the_probs[model_ind]
  inputs = tokenizer(" the", return_tensors='pt')['input_ids'].cuda()
  for pos_id in range(100):  # Check probability of "the the" for 100 positions
    probs = model(inputs, position_ids=torch.tensor([[pos_id]]).cuda())[
      'logits'][:, -1].softmax(-1)
    # Get probability "the" comes after "the"
    the_probs.append(probs[-1, [inputs[0][-1].item()]].item())

plt.rcParams["font.family"] = "Liberation Sans"
fig = plt.figure(figsize=(6, 3))
fig.suptitle("Sentence so far: 'the'", size=18)
ax1 = fig.add_subplot(1, 1, 1)

# Plot in reverse order so overlay puts pretrained on top.
for model_ind in range(len(models) - 1, -1, -1):
  ax1.scatter(np.arange(len(all_the_probs[model_ind])), 
              all_the_probs[model_ind], 
              label=model_names[model_ind])

ax1.set_xlabel("Position ID of 'the' ", size=16)
ax1.set_ylabel(r'$f($' + 'next word is also \'the\'' + r'$)$', size=16)
ax1.legend(loc='center right')

# Reorder legend so pretrained is on top.
handles, labels = ax1.get_legend_handles_labels()
handles = [handles[1], handles[0]]
labels = [labels[1], labels[0]]
ax1.legend(handles, labels, loc='center right')
plt.savefig(
  os.path.join(os.path.dirname(__file__), "figs/gpt2_the_repeats.pdf"), 
  dpi=300, 
  bbox_inches='tight')

