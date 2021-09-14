"""Methods to perform combinatorial rationalization in Huggingface."""
import itertools
import math
import time
import torch


def baseline_rationalize_lm(model,
                            input_ids,
                            tokenizer,
                            method,
                            verbose=True,
                            max_steps=1024,
                            start_step=0):
  """Perform a baseline search method to rationalize a language model.
  
  The objective is described in the `rationalize_lm` docstring. Rather than 
  maximizing the objective greedily, this function creates a global order of 
  tokens based on the sorting method. We add these tokens one at a time until 
  the combinatorial constraint is met.
  
  Args:
    model: A Huggingface model. The only requirement of this model is that it
      should accept a tensor of input tokens and a tensor of corresponding 
      positions and return a vector of logits corresponding to probabilities 
      over the vocabulary. The model should be trained or fine-tuned for 
      compatibility (i.e. with word dropout) to produce sensible rationales.
    input_ids: A tensor of shape `[seq_len]` containing the input IDs of the
      sequence to rationalize.
    tokenizer: A Huggingface `Tokenizer` object that tokenizes the vocabulary.
    method: A string describing which baseline method to use. Must be one of
      'gradient_norm', 'signed_gradient', 'integrated_gradient', 
      'attention_rollout', 'last_attention', or 'all_attention'.
    verbose: Whether to print out the rationalization results.
    max_steps: The maximum number of steps to perform search.
    start_step: The first token to rationalize. This function will rationalize
      the token at step `start_step` and continue to rationalize each token 
      until the end of the sentence. The function rationalizes the entire 
      sequence by default.
  
  Returns:
    all_rationales: A list of length `seq_len - 1` containing the tokens that
      form the rationales for the corresponding target tokens. The `t`-th entry
      of the list is the baseline rationale of the `t + 1`-st target token. The 
      list contains `seq_len - 1` rationales rather than `seq_len` rationales
      because we don't rationalize the first token of the sequence.
    log: A dictionary containing logging information from the rationalization 
      process.  
  """
  if method not in ['gradient_norm', 'signed_gradient', 'integrated_gradient',
                    'attention_rollout', 'last_attention', 'all_attention']:
    raise ValueError('Invalid method: {}'.format(method))
  all_rationales = []
  log = {}
  num_tokens = len(input_ids)
  start_time = time.time()

  input_text = [tokenizer.decode([token]) for token in input_ids]
  log['input_ids'] = list(input_ids.cpu().numpy())
  log['input_text'] = input_text
  log['rationalization'] = []

  if verbose:
    print("All tokens: {}".format(input_text))
  
  # Rationalize each token in the sequence, starting from `start_step`.
  for prev_token in range(start_step, num_tokens - 1):
    goal_word_id = input_ids[prev_token + 1]
    goal_word_text = input_text[prev_token + 1]
    token_log = {}
    token_log['target_position'] = prev_token + 1
    token_log['goal_word'] = goal_word_text
    token_log['log'] = []

    prev_tokens = input_ids[:prev_token + 1][None]
    # NOTE: we've modified modeling_gpt2.py so that if we pass in the input
    # embeddings, it assumes the positional encodings have already been added.
    # If we just use gradient norms rather than signed gradients, the two 
    # methods are identical because we're adding a constant. However they're
    # different for signed gradients because the embedding we multiply is 
    # different.
    initial_embeds = model.transformer.wte(prev_tokens)
    position_ids = torch.arange(len(prev_tokens[0]))[None].long().cuda()
    position_embeds = model.transformer.wpe(position_ids)
    embeddings = initial_embeds + position_embeds

    if method in ['gradient_norm', 'signed_gradient', 'integrated_gradient']:
      if method == 'integrated_gradient':
        # Approximate integrated gradient.
        path_integral_steps = 100
        all_gradients = []
        for i in range(0, path_integral_steps + 1):
          path_initial_embeds = initial_embeds * (i / path_integral_steps)
          path_full_embeds = path_initial_embeds + position_embeds
          path_logits = model(inputs_embeds=path_full_embeds)['logits']
          all_gradients.append(torch.autograd.grad(
            path_logits[:, -1].log_softmax(-1)[0, goal_word_id], 
            path_initial_embeds, 
            retain_graph=True)[0].detach())
        path_integral = torch.sum(torch.cat(all_gradients), 0)
        integrated_gradient = torch.sum(
          path_integral[None] / 
          (path_integral_steps + 1) * initial_embeds, -1)[0]
        integrated_gradient[-1] = float("inf")
        ordered_tokens = torch.argsort(-integrated_gradient)
      else:
        full_logits = model(inputs_embeds=embeddings)['logits']
        embedding_grad = torch.autograd.grad(
          full_logits[0, -1].log_softmax(-1)[goal_word_id], embeddings)
        if method == 'gradient_norm':
          grad_scores = embedding_grad[0].norm(dim=-1).view(-1)
        elif method == 'signed_gradient':
          grad_scores = torch.sum(embedding_grad[0] * embeddings, dim=-1)[0]
        # Make sure we always add the most recent token first.
        grad_scores[-1] = float("inf")
        ordered_tokens = torch.argsort(-grad_scores)
    else:
      # If we've reached here, we're doing an attention based ordering.
      outputs = model(inputs_embeds=embeddings, output_attentions=True)
      all_attentions = outputs['attentions']
      # Turn tuple into list, take mean over batch size (1) and number of heads
      attentions = torch.mean(torch.stack(all_attentions), [1, 2])
      num_layers, seq_len, _ = attentions.shape
      if method == 'attention_rollout':
        residualized_attentions = (
          0.5 * attentions + 0.5 * torch.eye(seq_len)[None].to(attentions))
        rollout = residualized_attentions[0]
        for layer in range(1, num_layers):
          rollout = torch.matmul(residualized_attentions[layer], rollout)
        rollout[-1, -1] = float("inf")
        ordered_tokens = torch.argsort(-rollout[-1])
      elif method == 'last_attention':
        last_attention = attentions[-1, -1]
        last_attention[-1] = float("inf")
        ordered_tokens = torch.argsort(-last_attention)
      else:
        # Average over all attention heads over all layers.
        all_attentions = torch.mean(attentions, 0)[-1]
        all_attentions[-1] = float("inf")
        ordered_tokens = torch.argsort(-all_attentions)
    
    rationale = []
    if verbose:
      print("Currently rationalizing token {}: '{}'".format(
        prev_token + 1, goal_word_text))
    
    for rationale_size in range(1, min(max_steps + 1, prev_token + 2)):
      full_candidate_rationale = ordered_tokens[:rationale_size]
      new_candidate = ordered_tokens[rationale_size - 1]
      rationale.append(new_candidate.item())

      added_token = input_text[new_candidate]
      added_token_string = "Adding token: '{}'".format(added_token)
      added_token_text = input_text[new_candidate]
      added_token_position = new_candidate.item()

      sorted_candidates = torch.sort(full_candidate_rationale).values
      candidate_input_ids = prev_tokens[:, sorted_candidates]
      candidate_positions = sorted_candidates[None]

      with torch.no_grad():
        logits = model(candidate_input_ids, 
                       position_ids=candidate_positions,)['logits'][0, -1]
        probs = logits.softmax(-1)
      predicted_word_id = logits.argmax().item()
      predicted_word_prob = probs.max().item()
      predicted_word_text = tokenizer.decode([predicted_word_id])
      true_token_prob = probs[input_ids[prev_token + 1]].item()
      token_log['log'].append({
        "rationale_size": rationale_size,
        "added_token_position": added_token_position,
        "added_token_text": added_token_text,
        "prediction": predicted_word_text,
        "prediction_prob": predicted_word_prob,
        "true_token_prob": true_token_prob,
      })
      if verbose:
        print("{}. This makes the top predicted word: '{}'. "
              "P('{}') = {:.3f}".format(
                added_token_string, predicted_word_text, 
                goal_word_text, true_token_prob))
      # Our combinatorial optimization is complete when the predicted token is
      # the true token.
      if torch.argmax(logits) == input_ids[prev_token + 1]:
        if verbose:
          print("When predicting: '{}'".format(goal_word_text))
          print("  The rationale is: {}".format(
            ', '.join([input_text[x] for x in rationale])))
          print("Finished with {} tokens.".format(rationale_size))
          print("..........")
        break
    # When we've finished rationalizing, add the rationale to the complete 
    # rationale list.
    all_rationales.append(rationale)
    token_log['rationale'] = rationale
    reached_argmax = predicted_word_id == input_ids[prev_token + 1]
    token_log['reached_argmax'] = reached_argmax.item()
    log['rationalization'].append(token_log)
  
  log['all_rationales'] = all_rationales
  end_time = time.time()
  log['duration'] = end_time - start_time
  return all_rationales, log
      

@torch.no_grad()
def exhaustive_rationalize_lm(model,
                              input_ids,
                              tokenizer,
                              verbose=True,
                              max_steps=6,
                              start_step=0,
                              max_tokens_per_batch=4096):
  """Exhaustively evaluate all possible rationales until one is sufficient.
  
  The objective is described in the `rationalize_lm` docstring. Rather than 
  maximizing the objective greedily, this function considers all possible 
  subsets of the input tokens, and returns the smallest subset that is 
  sufficient. Since there an exponential number of subsets, this method is
  quite expensive and in practice can only be used for very small rationales.
  
  Args:
    model: A Huggingface model. The only requirement of this model is that it
      should accept a tensor of input tokens and a tensor of corresponding 
      positions and return a vector of logits corresponding to probabilities 
      over the vocabulary. The model should be trained or fine-tuned for 
      compatibility (i.e. with word dropout) to produce sensible rationales.
    input_ids: A tensor of shape `[seq_len]` containing the input IDs of the
      sequence to rationalize.
    tokenizer: A Huggingface `Tokenizer` object that tokenizes the vocabulary.
    verbose: Whether to print out the rationalization results.
    max_steps: The largest subset size to consider. This function will 
      evaluate all subsets with size less than max_steps, and return the 
      smallest subset that is sufficient.
    start_step: The first token to rationalize. This function will rationalize
      the token at step `start_step` and continue to rationalize each token 
      until the end of the sentence. The function rationalizes the entire 
      sequence by default.
    max_tokens_per_batch: The maximum number of tokens to include for the
      batching step in greedy rationalization. This should depend on the size
      of the model. If you're getting model OOM errors, try making this 
      smaller.
  
  Returns:
    all_rationales: A list of length `seq_len - 1` containing the tokens that
      form the rationales for the corresponding target tokens. The `t`-th entry
      of the list is the baseline rationale of the `t + 1`-st target token. The 
      list contains `seq_len - 1` rationales rather than `seq_len` rationales
      because we don't rationalize the first token of the sequence.
    log: A dictionary containing logging information from the rationalization 
      process.  
  """
  all_rationales = []
  log = {}
  num_tokens = len(input_ids)
  start_time = time.time()

  input_text = [tokenizer.decode([token]) for token in input_ids]
  log['input_ids'] = list(input_ids.cpu().numpy())
  log['input_text'] = input_text
  log['rationalization'] = []

  if verbose:
    print("All tokens: {}".format(input_text))
  
  # Rationalize each token in the sequence, starting from `start_step`.
  for prev_token in range(start_step, num_tokens - 1):
    goal_word_id = input_ids[prev_token + 1]
    goal_word_text = input_text[prev_token + 1]
    token_log = {}
    token_log['target_position'] = prev_token + 1
    token_log['goal_word'] = goal_word_text
    token_log['log'] = []

    prev_tokens = input_ids[:prev_token + 1][None]
    optimum_found = False

    for rationale_size in range(1, prev_token + 2):
      if rationale_size > max_steps:
        if verbose:
          print("Didin't find any rationales with size <= {}".format(
            max_steps))
        log['all_rationales'] = None
        return None, log
      if verbose:
        print("Considering all rationales with size {}.".format(
          rationale_size))
      if rationale_size == 1:
        candidate_input_ids = prev_tokens[:, -1:]
        candidate_position_ids = torch.tensor([[prev_token]]).to(prev_tokens)
        best_logits = model(
          candidate_input_ids, 
          position_ids=candidate_position_ids)['logits'][0, -1]
        if torch.argmax(best_logits) == goal_word_id:
          best_subset = [prev_token]
          optimum_found = True
          break
      else:
        combinations = itertools.combinations(
          range(prev_token), rationale_size - 1)
        candidates = [sorted(list(combination) + [prev_token]) 
                      for combination in combinations]
        candidate_input_ids = prev_tokens[0, candidates]
        candidate_position_ids = torch.tensor(candidates).to(prev_tokens)

        num_subsets, seq_len = candidate_input_ids.size()
        batch_size = math.floor(max_tokens_per_batch / seq_len)
        num_batches = math.ceil(num_subsets / batch_size)
        for batch_ind in range(num_batches):
          batch_start_ind = batch_ind * batch_size
          batch_end_ind = (batch_ind + 1) * batch_size
          batch_input_ids = candidate_input_ids[batch_start_ind:batch_end_ind]
          batch_position_ids = candidate_position_ids[
            batch_start_ind:batch_end_ind]
          batch_logits = model(
            batch_input_ids, position_ids=batch_position_ids)['logits'][:, -1]
          if torch.any(batch_logits.argmax(-1) == goal_word_id):
            best_index = torch.where(
              batch_logits.argmax(-1) == goal_word_id)[0][0] + batch_start_ind
            best_subset = candidates[best_index]
            optimum_found = True
            break
      if optimum_found:
        if verbose:
          print("When predicting: '{}'".format(goal_word_text))
          print("  The rationale is: {}".format(
            ', '.join([input_text[x] for x in best_subset])))
          print("Finished with {} tokens.".format(rationale_size))
          print("..........")
        break
    all_rationales.append(best_subset)
  log['all_rationales'] = all_rationales
  end_time = time.time()
  log['duration'] = end_time - start_time
  return all_rationales, log


@torch.no_grad()
def rationalize_lm(model, 
                   input_ids, 
                   tokenizer, 
                   verbose=True, 
                   max_steps=1024, 
                   start_step=0,
                   max_tokens_per_batch=4096):
  """Perform greedy rationalization for a language model.

  For a given sequence y, the combinatorial objective for token t is given by:
    
    argmin_{s \in S} |s_y|
      s.t. argmax_{y_t'} p(y_t'|y_s) = y_t.

  Here, `S` refers to the set of all possible rationales (i.e. the power set
  containing all sequence token subsets), and `s` is an index set that 
  indicates sequence subsets that are considered as rationales. In other words, 
  the objective is to find the smallest subset of input tokens such that the 
  conditional probability of the `t`-th target token is maximized by the true 
  `t`-th target.

  This function maximizes this objective greedily. It begins with an empty 
  index set, and at each step adds the missing token that most maximizes the
  `t`-th target token. The procedure ends when the predicted token is the true
  target token.

  Args:
    model: A Huggingface model. The only requirement of this model is that it
      should accept a tensor of input tokens and a tensor of corresponding 
      positions and return a vector of logits corresponding to probabilities 
      over the vocabulary. The model should be trained or fine-tuned for 
      compatibility (i.e. with word dropout) to produce sensible rationales.
    input_ids: A tensor of shape `[seq_len]` containing the input IDs of the
      sequence to rationalize.
    tokenizer: A Huggingface `Tokenizer` object that tokenizes the vocabulary.
    verbose: Whether to print out the rationalization results.
    max_steps: The maximum number of steps to perform greedy rationalization.
    start_step: The first token to rationalize. This function will rationalize
      the token at step `start_step` and continue to rationalize each token 
      until the end of the sentence. The function rationalizes the entire 
      sequence by default.
    max_tokens_per_batch: The maximum number of tokens to include for the
      batching step in greedy rationalization. This should depend on the size
      of the model. If you're getting model OOM errors, try making this 
      smaller.
  
  Returns:
    all_rationales: A list of length `seq_len - 1` containing the tokens that
      form the rationales for the corresponding target tokens. The `t`-th entry
      of the list is the greedy rationale of the `t + 1`-st target token. The 
      list contains `seq_len - 1` rationales rather than `seq_len` rationales
      because we don't rationalize the first token of the sequence.
    log: A dictionary containing logging information from the rationalization 
      process.
  """
  all_rationales = []
  log = {}
  num_tokens = len(input_ids)
  start_time = time.time()

  input_text = [tokenizer.decode([token]) for token in input_ids]
  log['input_ids'] = list(input_ids.cpu().numpy())
  log['input_text'] = input_text
  log['rationalization'] = []

  if verbose:
    print("All tokens: {}".format(input_text))
  
  # Perform greedy rationalization for each token in the sequence, starting
  # from `start_step`.
  for prev_token in range(start_step, num_tokens - 1):
    goal_word_text = input_text[prev_token + 1]
    token_log = {}
    token_log['target_position'] = prev_token + 1
    token_log['goal_word'] = goal_word_text
    token_log['log'] = []
  
    # Initialize the rationale. The rationale must always include the most
    # recent token.
    rationale = [prev_token]

    if verbose:
      print("Currently rationalizing token {}: '{}'".format(
        prev_token + 1, goal_word_text))
    
    for rationale_size in range(1, min(max_steps + 1, prev_token + 2)):
      if rationale_size == 1:
        # A rationale of size 1 can only include the most recent target token.
        best_logits = model(
          input_ids[prev_token:(prev_token + 1)][None],
          position_ids=torch.tensor([[prev_token]]).to(input_ids))['logits']
        best_logits = best_logits[0, -1]
        added_token_text = input_text[prev_token]
        added_token_position = prev_token
        if verbose:
          added_token_string = ("Adding previous token to sequence: "
                                "'{}'".format(added_token_text))
      else:
        # Consider the current rationale + each target token
        candidates = [sorted(rationale + [x]) for x in range(prev_token + 1) 
                      if x not in rationale]
        candidate_input_ids = input_ids[[candidates]]
        candidate_position_ids = torch.tensor(candidates).to(input_ids)

        # Divide the candidates into batches, since all possible subsets may
        # not fit in memory if we pass them to the model at once.
        num_candidates, seq_len = candidate_input_ids.shape
        batch_size = math.floor(max_tokens_per_batch / seq_len)
        num_batches = math.ceil(num_candidates / batch_size)
        best_log_prob = -float("inf")
        for batch_ind in range(num_batches):
          batch_start_ind = batch_ind * batch_size
          batch_end_ind = (batch_ind + 1) * batch_size
          batch_input_ids = candidate_input_ids[batch_start_ind:batch_end_ind]
          batch_position_ids = candidate_position_ids[
            batch_start_ind:batch_end_ind]
          batch_logits = model(batch_input_ids, 
                               position_ids=batch_position_ids)['logits']
          # Only consider the logits for predicting the next token.
          batch_logits = batch_logits[:, -1]
          batch_log_probs = batch_logits.log_softmax(-1)[
            :, input_ids[prev_token + 1]]
          if batch_log_probs.max() > best_log_prob:
            best_log_prob = batch_log_probs.max()
            best_token = batch_log_probs.argmax() + batch_start_ind
            best_logits = batch_logits[batch_log_probs.argmax()]
        
        best_token_position = set(candidates[best_token]) - set(rationale)
        best_token_position = best_token_position.pop()
        rationale.append(best_token_position)
        added_token = input_text[best_token_position]
        added_token_string = "Adding token: '{}'".format(added_token)
        added_token_text = input_text[best_token_position]
        added_token_position = best_token_position
      
      best_probs = best_logits.softmax(-1)
      predicted_word_id = best_logits.argmax().item()
      predicted_word_prob = best_probs.max().item()
      predicted_word_text = tokenizer.decode([predicted_word_id])
      true_token_prob = best_probs[input_ids[prev_token + 1]].item()
      token_log['log'].append({
        "rationale_size": rationale_size,
        "added_token_position": added_token_position,
        "added_token_text": added_token_text,
        "prediction": predicted_word_text,
        "prediction_prob": predicted_word_prob,
        "true_token_prob": true_token_prob,
      })
      if verbose:
        print("{}. This makes the top predicted word: '{}'. "
              "P('{}') = {:.3f}".format(
                added_token_string, predicted_word_text, 
                goal_word_text, true_token_prob))
      # Our combinatorial optimization is complete when the predicted token is
      # the true token.
      if torch.argmax(best_logits) == input_ids[prev_token + 1]:
        if verbose:
          print("When predicting: '{}'".format(goal_word_text))
          print("  The rationale is: {}".format(
            ', '.join([input_text[x] for x in rationale])))
          print("Finished with {} tokens.".format(rationale_size))
          print("..........")
        break
    # When we've finished rationalizing, add the rationale to the complete 
    # rationale list.
    all_rationales.append(rationale)
    token_log['rationale'] = rationale
    reached_argmax = predicted_word_id == input_ids[prev_token + 1]
    token_log['reached_argmax'] = reached_argmax.item()
    log['rationalization'].append(token_log)
  
  log['all_rationales'] = all_rationales
  end_time = time.time()
  log['duration'] = end_time - start_time
  return all_rationales, log
