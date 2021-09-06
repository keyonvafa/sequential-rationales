"""Randomly concatenate pairs of translations for distractor experiment."""
import os
import numpy as np

project_dir = os.path.abspath(
  os.path.join(os.path.dirname(__file__), os.pardir))
fairseq_dir = os.path.join(project_dir, "fairseq")

# Load test source sentences.
test_de_file = os.path.join(
  fairseq_dir, "examples/translation/iwslt14.tokenized.de-en/test.de")
with open(test_de_file) as f:
  test_de = f.readlines()

# Load generated test translations.
test_en_file = os.path.join(
  fairseq_dir, "generated_translations/compatible_iwslt.txt")
with open(test_en_file) as f:
  test_en = f.readlines()

# Remove newlines from source and target sentences.
test_en = [line.rstrip('\n') for line in test_en]
test_de = [line.rstrip('\n') for line in test_de]

assert len(test_en) == len(test_de)
num_examples = len(test_en)

# Concatenate 500 random sentences to another set of 500 random sentences.
num_distractors = 500
rs = np.random.RandomState(0)
sampled_indices = rs.choice(num_examples, num_distractors * 2, replace=False)
first_half = sampled_indices[:num_distractors]
second_half = sampled_indices[num_distractors:]

# Add a space between the two sets of sentences.
distractors_en = [
  test_en[first_half[i]] + " " + test_en[second_half[i]] 
  for i in range(num_distractors)]
disractors_de = [
  test_de[first_half[i]] + " " + test_de[second_half[i]] 
  for i in range(num_distractors)]

# Keep track of the indices separating the two sets of sentences
breaks_en = [len(test_en[x].split(" ")) for x in first_half]
breaks_de = [len(test_de[x].split(" ")) for x in first_half]

# Save concatenated sentences and break indices.
distractor_dir = os.path.join(fairseq_dir, 'generated_translations')
with open(os.path.join(distractor_dir, 'distractors.en'), 'w') as f:
  for item in distractors_en:
    f.write("%s\n" % item)

with open(os.path.join(distractor_dir, 'distractors.de'), 'w') as f:
  for item in disractors_de:
    f.write("%s\n" % item)

with open(os.path.join(distractor_dir, 'breaks.en'), 'w') as f:
  for item in breaks_en:
    f.write("%s\n" % item)

with open(os.path.join(distractor_dir, 'breaks.de'), 'w') as f:
  for item in breaks_de:
    f.write("%s\n" % item)
