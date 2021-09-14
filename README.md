# Rationales for Sequential Predictions

## Quick start
Check out our [Colab notebook](https://colab.research.google.com/drive/1l33I0BDOXtPMdQVqB8Y24DJUp7K52qDz#scrollTo=KdN0dxky7nMw), which generates a sequence with GPT-2 and performs greedy rationalization. Our compatible version of GPT-2 is [available on Hugging Face](https://huggingface.co/keyonvafa/compatible-gpt2).

<p align="center">
<img src="https://github.com/keyonvafa/sequential-rationales/blob/main/analysis/figs/sequential_rationalization.gif" width="300" height="300" />
</p>

## Annotated Lambada
`annotated_lambada.json` is an annotated dataset based on [Lambada](https://arxiv.org/abs/1606.06031), containing 107 passages and their annotated rationales.  Each row has three keys: 
- `lambadaIndex` contains the corresponding (0-indexed) entry in Lambada.
- `text` contains the text of the full passage.
- `rationale`  contains the human rationales for predicting the final word of the passage. `rationale` is a list: each entry is a tuple of indices. The first index in each tuple represents the start of an annotation. The second index in each tuple represents the end of the corresponding annotation. The length of the list for each example is the size of its rationale.

To load the dataset with Pandas, run:
```python
import pandas as pd

df = pd.read_json('annotated_lambada.json', orient='records', lines=True)
# Print the rationale of the first example
text = df['text'].iloc[0]
rationale = df['rationale'].iloc[0]
print([text[sub_rationale[0]:sub_rationale[1]] for sub_rationale in rationale])
```

## Requirements and Installation
Configure a virtual environment using Python 3.6+ ([instructions here](https://docs.python.org/3.6/tutorial/venv.html)).
Inside the virtual environment, use `pip` to install the required packages:

```{bash}
pip install -r requirements.txt
```

Configure [Hugging Face](https://github.com/huggingface/transformers) to be developed locally
```{bash}
cd huggingface
pip install --editable ./
cd ..
```

Do the same with  [fairseq](https://github.com/pytorch/fairseq)
```{bash}
cd fairseq
pip install --editable ./
cd ..
```

Optionally, install NVIDIA's [apex](https://github.com/NVIDIA/apex) library to enable faster training
```{bash}
cd fairseq
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir \
  --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
cd ../..
```
All of the training commands assume a single GPU.

## Custom Datset

### Fine-tuning

TO DO

### Rationalization

TO DO

## Majority Class

### Preprocess
```{bash}
cd fairseq
TEXT=examples/language_model/majority_class
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train.tok \
    --validpref $TEXT/valid.tok \
    --testpref $TEXT/test.tok \
    --destdir data-bin/majority_class \
    --workers 20
```
### Train standard model and evaluate heldout perplexity
```{bash}
CHECKPOINT_DIR=/statlerdrive/keyonvafa/sequential-rationale-checkpoints
fairseq-train --task language_modeling \
    data-bin/majority_class \
    --arch transformer_lm_majority_class \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 --sample-break-mode eos \
    --max-tokens 64000 --update-freq 1 \
    --max-update 20000 \
    --no-epoch-checkpoints \
    --save-dir $CHECKPOINT_DIR/standard_majority_class \
    --tensorboard-logdir majority_class_logs/standard_majority_class \
    --word-dropout-mixture 0. --fp16 

fairseq-eval-lm data-bin/majority_class \
    --path $CHECKPOINT_DIR/standard_majority_class/checkpoint_best.pt \
    --batch-size 1024 \
    --tokens-per-sample 20 \
    --context-window 0
```
This should report 1.80 as the test set perplexity.

### Train compatible model
```{bash}
CHECKPOINT_DIR=/statlerdrive/keyonvafa/sequential-rationale-checkpoints
fairseq-train --task language_modeling \
    data-bin/majority_class \
    --arch transformer_lm_majority_class \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 --sample-break-mode eos \
    --max-tokens 64000 --update-freq 1 \
    --max-update 20000 \
    --no-epoch-checkpoints \
    --save-dir $CHECKPOINT_DIR/compatible_majority_class \
    --tensorboard-logdir majority_class_logs/compatible_majority_class \
    --word-dropout-mixture 1.0 --word-dropout-type uniform_length \
    --fp16 

fairseq-eval-lm data-bin/majority_class \
    --path $CHECKPOINT_DIR/compatible_majority_class/checkpoint_best.pt \
    --batch-size 1024 \
    --tokens-per-sample 20 \
    --context-window 0
```
This should also report 1.80 as the test set perplexity.

### Plot compatibility
```{bash}
cd ../analysis
python plot_majority_class_compatibility.py --checkpoint_dir $CHECKPOINT_DIR
cd ../fairseq
```

## IWSLT

### Download and preprocess the data
```{bash}
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

### Train standard transformer model
```{bash}
CHECKPOINT_DIR=/statlerdrive/keyonvafa/sequential-rationale-checkpoints
fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --max-tokens 4096 \
    --max-update 75000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir=iwslt_logs/standard_iwslt \
    --save-dir $CHECKPOINT_DIR/standard_iwslt \
    --no-epoch-checkpoints \
    --fp16 --word-dropout-mixture 0. 
```

### Copy standard transformer to new compatible folder
```bash
CHECKPOINT_DIR=/statlerdrive/keyonvafa/sequential-rationale-checkpoints
mkdir $CHECKPOINT_DIR/compatible_iwslt
cp $CHECKPOINT_DIR/standard_iwslt/checkpoint_best.pt $CHECKPOINT_DIR/compatible_iwslt/checkpoint_last.pt
```

### Train compatible transformer model
```{bash}
CHECKPOINT_DIR=/statlerdrive/keyonvafa/sequential-rationale-checkpoints
fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-5 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --max-tokens 4096 \
    --max-update 410000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --fp16 --reset-optimizer --reset-dataloader \
    --tensorboard-logdir=iwslt_logs/compatible_iwslt \
    --save-dir $CHECKPOINT_DIR/compatible_iwslt \
    --word-dropout-mixture 0.5  --word-dropout-type inverse_length \
    --no-epoch-checkpoints
```

### Evaluate BLEU
Standard:
```{bash}
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $CHECKPOINT_DIR/standard_iwslt/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```
This should report ~34.76

Compatible:
```{bash}
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $CHECKPOINT_DIR/compatible_iwslt/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```
This should report ~34.78.

### Generate translations for distractor experiment
```{bash}
mkdir generated_translations
fairseq-generate data-bin/iwslt14.tokenized.de-en \
  --path $CHECKPOINT_DIR/compatible_iwslt/checkpoint_best.pt \
  --batch-size 128 \
  --beam 1  > generated_translations/compatible_iwslt_tmp.txt
grep 'H-' \
  generated_translations/compatible_iwslt_tmp.txt \
  | sed 's/^..//' | sort -n | \
  cut -f3 > generated_translations/compatible_iwslt.txt
```

### Randomly concatenate generated sentences and binarize
```{bash}
cd ../analysis
python create_distractor_iwslt_dataset.py
cd ../fairseq
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train  \
    --validpref $TEXT/valid \
    --testpref generated_translations/distractors \
    --destdir data-bin/iwslt14_distractors.tokenized.de-en \
    --workers 20
```

### Perform greedy rationalization for distractor dataset
```{bash}
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method greedy
```

### Perform baseline rationalization for distractor dataset
```{bash}
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method gradient_norm
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method signed_gradient
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method integrated_gradient
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method last_attention
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task distractors  --method all_attention
```
And repeat for method in gradient_norm, signed_gradient, integrated_gradient, last_attention, all_attention

### Evaluate distractors
```{bash}
CHECKPOINT_DIR=/statlerdrive/keyonvafa/sequential-rationale-checkpoints
cd ../analysis
python evaluate_distractor_rationales.py --baseline gradient_norm
python evaluate_distractor_rationales.py --baseline signed_gradient
python evaluate_distractor_rationales.py --baseline integrated_gradient
python evaluate_distractor_rationales.py --baseline last_attention
python evaluate_distractor_rationales.py --baseline all_attention
cd ../fairseq
```
|       Method      | Source Mean | Target Mean | Source Frac. | Target Frac. |
| ----------------- | ----------- | ----------- | ------------ | ------------ |
| Gradient norms    |     0.40    |     0.44    |   **0.06**   |     0.06     |
| Grad x emb        |     6.25    |     5.57    |     0.42     |     0.41     |
| Integrated grads  |     2.08    |     1.68    |     0.23     |     0.14     |
| Last attention    |     0.63    |     2.41    |     0.09     |     0.24     |
| All attentions    |     0.58    |     0.80    |     0.08     |     0.12     |
| Greedy            |   **0.12**  |   **0.12**  |     0.09     |   **0.02**   |

Results for `checkpoint_best` of 410K steps:


### Download and preprocess alignments
First, agree to the license download the gold alignments from [RWTH Aachen](https://www-i6.informatik.rwth-aachen.de/goldAlignment/). Put the files `en`, `de`, and `alignmentDeEn` in the directory `fairseq/examples/translation/iwslt14.tokenized.de-en/gold_labels`, and, in that same repo, convert to Unicode using
```{bash}
iconv -f ISO_8859-1 -t UTF8 de > gold.de
iconv -f ISO_8859-1 -t UTF8 en > gold.en
```

Clean and tokenize the text
```{bash}
cd ../..
cat iwslt14.tokenized.de-en/gold_labels/gold.en | \
  perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -l en | \
  perl mosesdecoder/scripts/tokenizer/lowercase.perl > iwslt14.tokenized.de-en/gold_labels/tmp_gold.en
cat iwslt14.tokenized.de-en/gold_labels/gold.de | \
  perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -l de | \
  perl mosesdecoder/scripts/tokenizer/lowercase.perl > iwslt14.tokenized.de-en/gold_labels/tmp_gold.de
cd ../..
```

Apply BPE

```{bash}
python subword-nmt/subword_nmt/apply_bpe.py -c iwslt14.tokenized.de-en/code < iwslt14.tokenized.de-en/gold_labels/tmp_gold.en > iwslt14.tokenized.de-en/gold_labels/gold_bpe.en 
python subword-nmt/subword_nmt/apply_bpe.py -c iwslt14.tokenized.de-en/code < iwslt14.tokenized.de-en/gold_labels/tmp_gold.de > iwslt14.tokenized.de-en/gold_labels/gold_bpe.de
```

Since the original file automatically tokenizes the apostrophes (e.g. `don ' t`) after BPE, sometimes there are incorrect spaces in the tokenization (e.g. `don &apos; t` instead of `don &apos;t`). Since this would change the alignments for these files, you may need to go through the files `gold_bpe.en` and `gold_bpe.de` and change them manually. Keep the spaces for apostrophes but not for plurals, e.g. `man &apos;s office` and `&apos; legal drugs &apos; (` are both correct. Delete the new lines at the bottom of `gold.en`, `gold.de`, `gold_bpe.en`, and `gold_bpe.de`. The only other necessary change is changing `à@@ -@@` to `a-@@` on line 247 since we don't tokenize accents.

If you would like to skip these steps, email me at [keyvafa@gmail.com](mailto:keyvafa@gmail.com) and I can provide you the preprocessed files.

### Binarize gold alignments
```{bash}
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train  \
    --validpref $TEXT/valid \
    --testpref $TEXT/gold_labels/gold_bpe \
    --destdir data-bin/iwslt14_alignments.tokenized.de-en \
    --workers 20
```

### Create mapping between alignments with/without BPE
```{bash}
CHECKPOINT_DIR=/statlerdrive/keyonvafa/sequential-rationale-checkpoints
cd ../analysis
python map_alignments_to_bpe.py --checkpoint_dir $CHECKPOINT_DIR
cd ../fairseq
```

### Perform greedy rationalization for alignments dataset
```{bash}
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method greedy
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method greedy  --top_1
```

### Perform baseline rationalization for alignments dataset
```{bash}
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method gradient_norm
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method gradient_norm  --top_1
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method signed_gradient
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method signed_gradient  --top_1
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method integrated_gradient
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method integrated_gradient  --top_1
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method last_attention
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method last_attention  --top_1
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method all_attention
python rationalize_iwslt.py --checkpoint_dir $CHECKPOINT_DIR \
    --task alignments  --method all_attention  --top_1
```
Repeat the above two steps with the `--top_1` flag, and also for each baseline.

### Evaluate alignments
```{bash}
cd ../analysis
python evaluate_alignment_rationales.py --baseline gradient_norm 
python evaluate_alignment_rationales.py --baseline gradient_norm  --top_1
python evaluate_alignment_rationales.py --baseline signed_gradient 
python evaluate_alignment_rationales.py --baseline signed_gradient  --top_1
python evaluate_alignment_rationales.py --baseline integrated_gradient 
python evaluate_alignment_rationales.py --baseline integrated_gradient  --top_1
python evaluate_alignment_rationales.py --baseline last_attention 
python evaluate_alignment_rationales.py --baseline last_attention  --top_1
python evaluate_alignment_rationales.py --baseline all_attention 
python evaluate_alignment_rationales.py --baseline all_attention  --top_1
cd ../fairseq
```
And repeat with each baseline.

|       Method       | Length |  AER   |  IOU   |   F1   |  Top1  |
| ------------------ | ------ | ------ | ------ | ------ | ------ |
| Gradient norms     |  10.2  |  0.82  |  0.30  |  0.16  |  0.63  |
| Grad x emb         |  13.2  |  0.90  |  0.16  |  0.12  |  0.40  |
| Integrated grads   |  11.3  |  0.85  |  0.24  |  0.14  |  0.42  |
| Last attention     |  10.8  |  0.84  |  0.27  |  0.15  |  0.59  |
| All attentions     |  10.7  |  0.82  |  0.32  |  0.15  |**0.66**|
| Greedy             | **4.9**|**0.78**|**0.40**|**0.24**|  0.64  |


### Plot greedy rationalization example from Fairseq
```{bash}
python plot_iwslt_rationalization.py
cd ..
```

## GPT-2

In the paper, we performed experiments for fine-tuning GPT-2 Large (using sequence lengths of 1024). Since practitioners may not have a GPU that has the memory capacity to train the large model, our replication instructions are for GPT-2 Medium, fine-tuning with a sequence length of 512. This can be done on a single 12GB GPU, and the rationalization performance is similar for both models. If you would like to specifically replicate our results for GPT-2 Large, email me at [keyvafa@gmail.com](mailto:keyvafa@gmail.com) and I can provide you the fine-tuning instructions/the full fine-tuned model.

### Download Open-Webtext

First go to the [OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/), a WebText replication corpus provided by Aaron Gokaslan and Vanya Cohen. We only use a single split to train (we used `urlsf_subset09.tar`). Expand all the items and merge the first 998, taking only the first 8 million lines. This will be the training set. We used half of the remaining files as the validation set, and the other half as the test set. Store the files as `webtext_train.txt`, `webtext_valid.txt`, and `webtext_test.txt` in `huggingface/data`.

Alternatively, you can email me at [keyvafa@gmail.com](mailto:keyvafa@gmail.com) and I can send you the raw files (they're a little too large to store on Github).

### Fine-tune GPT-2 for compatibility using word dropout
Note about Apex....
```{bash}
cd huggingface
CHECKPOINT_DIR=/statlerdrive/keyonvafa/sequential-rationale-checkpoints
python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path gpt2-medium \
    --do_train \
    --do_eval \
    --train_file data/webtext_train.txt \
    --validation_file data/webtext_valid.txt \
    --logging_dir gpt2_logs/compatible_gpt2 \
    --output_dir $CHECKPOINT_DIR/compatible_gpt2 \
    --per_device_train_batch_size 1 \
    --evaluation_strategy steps --eval_steps 500 \
    --num_train_epochs 50 \
    --lr_scheduler_type constant \
    --learning_rate 0.00001 \
    --block_size 512 \
    --per_device_eval_batch_size 4 \
    --save_total_limit 2 \
    --max_steps 45000 \
    --word_dropout_mixture 0.5
```

### Get heldout perplexity
For the compatible model:
```{bash}
python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $CHECKPOINT_DIR/compatible_gpt2 \
    --output_dir gpt2_test_output/compatible/ \
    --do_eval \
    --validation_file data/webtext_test.txt \
    --block_size 512 \
    --per_device_eval_batch_size 4
```
This should give a heldout perplexity of 17.6086.

For the pretrained model:
```{bash}
python examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path gpt2-medium \
    --output_dir gpt2_test_output/pretrained/ \
    --do_eval \
    --validation_file data/webtext_test.txt \
    --block_size 512 \
    --per_device_eval_batch_size 4
```
This should give a heldout perplexity of 19.9674.

### Plot "the" repeats to check compatibility
```{bash}
cd ../analysis
python plot_gpt2_the_repeats.py  --checkpoint_dir $CHECKPOINT_DIR
cd ../huggingface
```

### Plot page 1 figure
```{bash}
python plot_gpt2_rationalization.py  --checkpoint_dir $CHECKPOINT_DIR
```

### Greedy rationalize analogies
```{bash}
python rationalize_analogies.py  --checkpoint_dir $CHECKPOINT_DIR  \
    --method greedy 
```

### Greedy rationalize baselines and exhaustive
```{bash}
python rationalize_analogies.py  --checkpoint_dir $CHECKPOINT_DIR  \
    --method gradient_norm
python rationalize_analogies.py  --checkpoint_dir $CHECKPOINT_DIR  \
    --method signed_gradient
python rationalize_analogies.py  --checkpoint_dir $CHECKPOINT_DIR  \
    --method integrated_gradient
python rationalize_analogies.py  --checkpoint_dir $CHECKPOINT_DIR  \
    --method attention_rollout
python rationalize_analogies.py  --checkpoint_dir $CHECKPOINT_DIR  \
    --method all_attention
python rationalize_analogies.py  --checkpoint_dir $CHECKPOINT_DIR  \
    --method last_attention
```

### Evaluate rationales for analogies experiment
```{bash}
cd ../analysis
python evaluate_analogies_rationales.py --baseline gradient_norm
python evaluate_analogies_rationales.py --baseline signed_gradient
python evaluate_analogies_rationales.py --baseline integrated_gradient
python evaluate_analogies_rationales.py --baseline attention_rollout
python evaluate_analogies_rationales.py --baseline last_attention
python evaluate_analogies_rationales.py --baseline all_attention
cd ../huggingface
```

|       Method       | Length | Ratio |  Ante  |  No D  |
| ------------------ | ------ | ----- | ------ | ------ |
| Gradient norms     |  14.9  |  2.7  |**1.00**|  0.22  |
| Grad x emb         |  33.1  |  6.0  |  0.99  |  0.10  |
| Integrated grads   |  31.9  |  6.0  |  0.99  |  0.04  |
| Attention rollout  |  37.6  |  7.0  |**1.00**|  0.04  |
| Last attention     |  14.9  |  2.7  |**1.00**|  0.31  |
| All attentions     |  11.2  |  2.3  |  0.99  |  0.32  |
| Greedy             | **7.8**|**1.1**|**1.00**|**0.63**|



### Get wall-clock time comparisons
```{bash}
python compare_rationalization_times.py  --checkpoint_dir $CHECKPOINT_DIR
```

### Greedily rationalize Lambada
```{bash}
python rationalize_annotated_lambada.py --checkpoint_dir $CHECKPOINT_DIR \
    --method greedy 
```

### Rationalize Lambada baselines
```{bash}
python rationalize_annotated_lambada.py --checkpoint_dir $CHECKPOINT_DIR \
    --method gradient_norm
python rationalize_annotated_lambada.py --checkpoint_dir $CHECKPOINT_DIR \
    --method signed_gradient
python rationalize_annotated_lambada.py --checkpoint_dir $CHECKPOINT_DIR \
    --method integrated_gradient
python rationalize_annotated_lambada.py --checkpoint_dir $CHECKPOINT_DIR \
    --method attention_rollout
python rationalize_annotated_lambada.py --checkpoint_dir $CHECKPOINT_DIR \
    --method last_attention
python rationalize_annotated_lambada.py --checkpoint_dir $CHECKPOINT_DIR \
    --method all_attention 
```

### Evaluate rationales for Lambada
```{bash}
cd ../analysis
python evaluate_lambada_rationales.py --baseline gradient_norm
python evaluate_lambada_rationales.py --baseline signed_gradient
python evaluate_lambada_rationales.py --baseline integrated_gradient
python evaluate_lambada_rationales.py --baseline attention_rollout
python evaluate_lambada_rationales.py --baseline last_attention
python evaluate_lambada_rationales.py --baseline all_attention
cd ../huggingface
```

|       Method       | Length |  IOU   |   F1   |
| ------------------ | ------ | ------ | ------ |
| Gradient norms     |  53.4  |  0.17  |  0.25  |
| Grad x emb         |  66.6  |  0.13  |  0.21  |
| Integrated grads   |  67.0  |  0.12  |  0.21  |
| Attention rollout  |  72.6  |  0.10  |  0.18  |
| Last attention     |  53.0  |  0.16  |  0.25  | 
| All attentions     |  51.3  |  0.19  |  0.26  | 
| Greedy             |**18.9**|**0.25**|**0.34**| 

