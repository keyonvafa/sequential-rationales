"""Tools to perform combinatorial rationalization in Fairseq."""
import math
import torch

from fairseq import utils


def baseline_rationalize_conditional_model(model,
                                           source_tokens,
                                           target_tokens,
                                           method,
                                           verbose=True,
                                           max_steps=1024,
                                           top_1=False):
  """Perform a baseline search method to rationalize a conditional model.

  The objective is described in the `rationalize_conditional_model` docstring. 
  Rather than maximizing the objective greedily, we create a global order of 
  tokens based on the sorting method. We add these tokens one at a time until 
  the combinatorial constraint is met.

  Args:
    model: A Fairseq model. It should be a `GeneratorHubInterface` object, with
      `model.model` as an object that implements the 
      `FairseqEncoderDecoderModel` interface. The model should be trained or
      fine-tuned for compatibility (i.e. with word dropout) to produce
      sensible rationales.
    source_tokens: A tensor of shape `[src_len]` containing the source tokens.
    target_tokens: A tensor of shape `[tgt_len]` containing the target tokens.
    method: A string describing which baseline method to use. Must be one of
      'gradient_norm', 'signed_gradient', 'integrated_gradient', 
      'last_attention', or 'all_attention'.
    verbose: Whether to print out the rationalization results.
    max_steps: The maximum number of steps to perform greedy rationalization.
    top_1: Whether to stop after including a source token in the rationale.
  
  Returns:
    all_source_rationales: A list of length `tgt_len` containing the source 
      tokens that form the best rationales for the corresponding target tokens. 
      The `t`-th entry of the list is a list of source tokens found for the 
      greedy rationale of the `t`-th target token.
    all_target_rationales: A list of length `tgt_len` containing the target 
      tokens that form the best rationales for the corresponding target tokens. 
      The `t`-th entry of the list is a list of target tokens found for the 
      greedy rationale of the `t`-th target token.
    log: A dictionary containing logging information from the search process.
  """
  if method not in ['gradient_norm', 'signed_gradient', 'integrated_gradient',
                    'last_attention', 'all_attention']:
    raise ValueError('Invalid method: {}'.format(method))
  
  all_source_rationales = []
  all_target_rationales = []
  log = {}

  num_source_tokens = len(source_tokens)
  num_target_tokens = len(target_tokens)

  source_tokens_text = decode_sequence(model, source_tokens, source=True)
  target_tokens_text = decode_sequence(model, target_tokens, source=False)
  log['source_tokens'] = list(source_tokens.cpu().numpy())
  log['target_tokens'] = list(target_tokens.cpu().numpy())
  log['source_tokens_text'] = source_tokens_text
  log['target_tokens_text'] = target_tokens_text
  log['rationalizations'] = []

  if verbose:
    print('All source tokens: {}'.format(source_tokens_text))
    print('All target tokens: {}'.format(target_tokens_text))

  # Get position IDs for all tokens.
  all_source_positions = utils.make_positions(
    source_tokens[None], model.model.encoder.padding_idx)
  all_target_positions = utils.make_positions(
    target_tokens[None], model.model.decoder.padding_idx)
    
  return_embeddings = method in [
    'gradient_norm', 'signed_gradient', 'integrated_gradient']
  return_all_attns = method in ['last_attention', 'all_attention']

  full_encoder_out = model.model.encoder(
    source_tokens[None], 
    return_embeddings=return_embeddings, 
    return_all_attns=return_all_attns)
  full_decoder_out = model.model.decoder(
    prev_output_tokens=target_tokens[None],
    encoder_out=full_encoder_out,
    return_embeddings=return_embeddings,
    return_all_attns=return_all_attns)
  
  for target_token in range(num_target_tokens - 1):
    goal_word_text = target_tokens_text[target_token + 1]
    goal_word_id = target_tokens[target_token + 1]
    token_log = {}
    token_log['target_position'] = target_token + 1
    token_log['goal_word'] = goal_word_text
    token_log['log'] = []

    lprobs = model.model.get_normalized_probs(
      full_decoder_out, log_probs=True)[0, target_token]
    
    if method in ['gradient_norm', 'signed_gradient']:
      encoder_embedding = full_encoder_out['first_layer_embedding'][0]
      decoder_embedding = full_decoder_out[1]['first_layer_embedding'][0]
      # Only consider gradients before the most recent target token.
      encoder_grad = torch.autograd.grad(
        lprobs[goal_word_id], encoder_embedding, retain_graph=True)[0][0]
      decoder_grad = torch.autograd.grad(
        lprobs[goal_word_id], decoder_embedding, retain_graph=True)[0]
      # Can't get gradients from after the most recent target token.
      decoder_grad = decoder_grad[0, :(target_token + 1)]
      decoder_embedding = decoder_embedding[:, :(target_token + 1)]
    
    if method == 'gradient_norm':
      encoder_grad_norm = torch.norm(encoder_grad, dim=1)
      decoder_grad_norm = torch.norm(decoder_grad, dim=1)
      concatenated_tokens = torch.cat([encoder_grad_norm, decoder_grad_norm])
      # Make sure we always add the most recent token first.
      concatenated_tokens[-1] = float("inf")
      ordered_tokens = torch.argsort(-concatenated_tokens)
    elif method == 'signed_gradient':
      # `embedding_grad` is gradient of negative loss, so don't take negative.
      encoder_grad_norm = torch.sum(encoder_grad * encoder_embedding, -1)[0]
      decoder_grad_norm = torch.sum(decoder_grad * decoder_embedding, -1)[0]
      concatenated_tokens = torch.cat([encoder_grad_norm, decoder_grad_norm])
      # Make sure we always add the most recent token first.
      concatenated_tokens[-1] = float("inf")
      ordered_tokens = torch.argsort(-concatenated_tokens)
    elif method == 'integrated_gradient':
      encoder_embedding = full_encoder_out['first_layer_embedding'][0]
      input_encoder_embedding = (
        (encoder_embedding - 
         model.model.encoder.embed_positions(source_tokens[None])
         ) / model.model.encoder.embed_scale).detach().requires_grad_(True)
      # Approximate integrated gradient with path integral.
      # See https://arxiv.org/abs/1703.01365.
      num_path_steps = 50
      path_encoder_embeddings = torch.stack(
        [input_encoder_embedding * i / num_path_steps 
         for i in range(num_path_steps + 1)])[:, 0]
      
      # Do the same for the decoder
      decoder_embedding = full_decoder_out[1]['first_layer_embedding'][0][
        :, :(target_token + 1)]
      input_decoder_embedding = (
        (decoder_embedding - model.model.decoder.embed_positions(
          target_tokens[:(target_token + 1)][None])
         ) / model.model.decoder.embed_scale).detach().requires_grad_(True)
      path_decoder_embeddings = torch.stack(
        [input_decoder_embedding * i / num_path_steps 
         for i in range(num_path_steps + 1)])[:, 0]
      
      # Pass into encoder and decoder.
      integrated_encoder_out = model.model.encoder(
        source_tokens[None].repeat(num_path_steps + 1, 1),
        return_embeddings=True,
        token_embeddings=path_encoder_embeddings,)
      integrated_decoder_out = model.model.decoder(
        prev_output_tokens=target_tokens[: (target_token + 1)][None].repeat(
          [num_path_steps + 1, 1]), 
        encoder_out=integrated_encoder_out, 
        token_embeddings=path_decoder_embeddings,)
      integrated_logits = integrated_decoder_out[0][:, -1]

      # Sum over all path steps.
      all_encoder_grads = torch.stack(
        [torch.autograd.grad(
          integrated_logits.log_softmax(-1)[i, goal_word_id], 
          path_encoder_embeddings, retain_graph=True)[0][i] 
         for i in range(num_path_steps + 1)])
      all_decoder_grads = torch.stack(
        [torch.autograd.grad(
          integrated_logits.log_softmax(-1)[i, goal_word_id],
          path_decoder_embeddings, retain_graph=True)[0][i] 
         for i in range(num_path_steps + 1)])
      encoder_path_integral = torch.sum(all_encoder_grads, 0)
      decoder_path_integral = torch.sum(all_decoder_grads, 0)
      encoder_integrated_gradient = torch.sum(
        encoder_embedding / (num_path_steps + 1) * 
        encoder_path_integral[None], -1)[0]
      decoder_integrated_gradient = torch.sum(
        decoder_embedding / (num_path_steps + 1) * 
        decoder_path_integral[None], -1)[0]
      concatenated_tokens = torch.cat(
        [encoder_integrated_gradient, decoder_integrated_gradient])
      
      # Make sure the most recent token is always first.
      concatenated_tokens[-1] = float("inf")
      ordered_tokens = torch.argsort(-concatenated_tokens)
    elif method == 'last_attention':
      decoder_self_attns = torch.stack(full_decoder_out[1]['self_attns'], 0)
      decoder_cross_attns = torch.stack(full_decoder_out[1]['cross_attns'], 0)
      # Can't get attention from tokens after the most recent target token.
      decoder_self_attns = decoder_self_attns[:, :, :, :(target_token + 1)]
      # Take the last layer attention weights.
      decoder_self_attns = decoder_self_attns[-1, 0, target_token]
      decoder_cross_attns = decoder_cross_attns[-1, 0, target_token]
      concatenated_tokens = torch.cat(
        [decoder_cross_attns, decoder_self_attns])
      # Make sure we always add the most recent token first.
      concatenated_tokens[-1] = float("inf")
      ordered_tokens = torch.argsort(-concatenated_tokens)
    elif method == 'all_attention':
      decoder_self_attns = torch.stack(full_decoder_out[1]['self_attns'], 0)
      decoder_cross_attns = torch.stack(full_decoder_out[1]['cross_attns'], 0)
      # Can't get attention from tokens after the most recent target token.
      decoder_self_attns = decoder_self_attns[:, :, :, :(target_token + 1)]
      # Make sure we've already averaged over heads.
      assert decoder_cross_attns.shape[1] == 1
      assert decoder_self_attns.shape[1] == 1
      # Average over layers
      decoder_self_attns = torch.mean(decoder_self_attns, 0)[0, target_token]
      decoder_cross_attns = torch.mean(decoder_cross_attns, 0)[0, target_token]
      concatenated_tokens = torch.cat(
        [decoder_cross_attns, decoder_self_attns])
      # Make sure we always add the most recent token first.
      concatenated_tokens[-1] = float("inf")
      ordered_tokens = torch.argsort(-concatenated_tokens)
    
    if verbose:
      print("Currently rationalizing token {}: '{}'".format(
        target_token + 1, 
        goal_word_text))
    
    max_possible_steps = target_token + num_source_tokens + 2
    max_rationale_size = min(max_steps + 1, max_possible_steps)
    for rationale_size in range(1, max_rationale_size):
      full_candidate_rationale = ordered_tokens[:rationale_size]
      new_candidate = ordered_tokens[rationale_size - 1]
      if new_candidate < num_source_tokens:
        added_token_position = new_candidate
        added_token_from = "source"
        added_token_text = source_tokens_text[new_candidate]
      else:
        added_token_position = new_candidate - num_source_tokens
        added_token_from = "target"
        added_token_text = target_tokens_text[
          new_candidate - num_source_tokens]
      unsorted_source_rationale = full_candidate_rationale[
        full_candidate_rationale < num_source_tokens]
      unsorted_target_rationale = full_candidate_rationale[
        full_candidate_rationale >= num_source_tokens] - num_source_tokens
      # Sort the kept rationales by position so we don't interfere with
      # autoregressive masking in the decoder.
      source_rationale = torch.sort(unsorted_source_rationale).values
      target_rationale = torch.sort(unsorted_target_rationale).values
      
      # If there are no source tokens in our rationale, pass an empty input to
      # the encoder.
      if len(source_rationale) == 0:
        encoder_out = model.model.encoder(
          torch.tensor([[model.model.encoder.padding_idx]]).to(model.device))
      else:
        encoder_src_tokens = source_tokens[source_rationale][None]
        encoder_positions = all_source_positions[0, source_rationale][None]
        encoder_out = model.model.encoder(encoder_src_tokens,
                                          position_ids=encoder_positions)

      decoder_input_tokens = target_tokens[target_rationale][None]
      decoder_positions = all_target_positions[0, target_rationale][None]
      decoder_out = model.model.decoder(
        decoder_input_tokens, 
        encoder_out=encoder_out,
        position_ids=decoder_positions)
      probs = model.model.get_normalized_probs(
        decoder_out, log_probs=False)[:, -1]
      predicted_word_id = probs.argmax().item()
      predicted_word_prob = probs.max().item()
      predicted_word_text = decode_single_word(model, predicted_word_id, 
                                               source=False)
      true_target_prob = probs[:, target_tokens[target_token + 1]].item()
      
      if new_candidate < num_source_tokens:
        added_token_string = "Adding source token: '{}'.".format(
          source_tokens_text[new_candidate])
      else:
        added_token_string = "Adding target token: '{}'.".format(
          target_tokens_text[new_candidate - num_source_tokens])
      if verbose:
        print("{} This makes the top predicted word: '{}'. "
              "P('{}') = {:.3f}".format(
                added_token_string, predicted_word_text, 
                goal_word_text, true_target_prob))

      token_log['log'].append({
        "rationale_size": rationale_size,
        "added_token_position": added_token_position.item(),
        "added_token_text": added_token_text,
        "from": added_token_from,
        "prediction": predicted_word_text,
        "prediction_prob": predicted_word_prob,
        "true_token_prob": true_target_prob,
      })

      # Search is done if the predicted token is the goal token, or if we're
      # doing top-1 rationalization and we've added a source token. We finish
      # if we've added 2 source tokens rather than 1 in case the first added
      # token is <eos>.
      if (
        (predicted_word_id == target_tokens[target_token + 1] and not top_1) or
        (top_1 and len(source_rationale) >= 2)):
        if verbose:
          print("When predicting: '{}'".format(goal_word_text))
          print("  The source rationale is: {}".format(
            ', '.join([source_tokens_text[x] for x in source_rationale])))
          print("  The target rationale is: {}".format(
            ', '.join([target_tokens_text[x] for x in target_rationale])))
          print("Finished with {} tokens: {} in source and "
                "{} in target.".format(rationale_size, 
                                       len(source_rationale), 
                                       len(target_rationale)))
          print("..........")
        break
    # Add the unsorted rationales, since these contain the original ordering
    # (i.e. by baseline importance scores) rather than sorted by position.
    all_source_rationales.append(list(unsorted_source_rationale.cpu().numpy()))
    all_target_rationales.append(list(unsorted_target_rationale.cpu().numpy()))
    token_log['source_rationale'] = list(
      unsorted_source_rationale.cpu().numpy())
    token_log['target_rationale'] = list(
      unsorted_target_rationale.cpu().numpy())
    reached_argmax = predicted_word_id == target_tokens[target_token + 1]
    token_log['reached_argmax'] = reached_argmax.item()
    log['rationalizations'].append(token_log)
  
  log['all_source_rationales'] = all_source_rationales
  log['all_target_rationales'] = all_target_rationales
  return all_source_rationales, all_target_rationales, log


def decode_single_word(model, word_index, source=False):
  if source:
    dictionary = model.task.source_dictionary
  else:
    dictionary = model.task.target_dictionary
  if model.bpe is None:
    return dictionary.symbols[word_index]
  return model.bpe.decode(dictionary.string([word_index]))


def decode_sequence(model, tokens, source=False):
  return [decode_single_word(model, token, source) for token in tokens]


@torch.no_grad()
def rationalize_conditional_model(model,
                                  source_tokens, 
                                  target_tokens, 
                                  verbose=True,
                                  max_steps=1024,
                                  top_1=False):
  """Perform greedy rationalization for a conditional model.

  For a given source sequence `x` and target sequence `y` up to token `t - 1`,
  the combinatorial objective is:
    
    argmin_{S_x, S_y \in S} |S_x| + |S_y|
      s.t. argmax_{y_t'} p(y_t'|x_{S_x}, y_{S_y}) = y_t.

  Here, `S` refers to the set of all possible rationales (i.e. the cross 
  product of the source and target sequence power sets), and `S_x` and `S_y` 
  are index sets that indicate the source and target sequence subsets that are 
  considered as rationales. In other words, the objective is to find the
  smallest set of source and target tokens such that the conditional
  probability of the `t`-th target token is maximized by the true `t`-th
  target.

  This function maximizes this objective greedily. It begins with empty index
  sets, and at each step adds the missing token that most maximizes the
  `t`-th target token. The procedure ends when the predicted token is the true
  target token.

  Args:
    model: A Fairseq model. It should be a `GeneratorHubInterface` object, with
      `model.model` as an object that implements the 
      `FairseqEncoderDecoderModel` interface. The model should be trained or
      fine-tuned for compatibility (i.e. with word dropout) to produce
      sensible rationales.
    source_tokens: A tensor of shape `[src_len]` containing the source tokens.
    target_tokens: A tensor of shape `[tgt_len]` containing the target tokens.
    verbose: Whether to print out the rationalization results.
    max_steps: The maximum number of steps to perform greedy rationalization.
    top_1: Whether to stop after including a source token in the rationale.
  
  Returns:
    all_source_rationales: A list of length `tgt_len - 1` containing the source 
      tokens that form the best rationales for the corresponding target tokens. 
      The `t`-th entry of the list is a list of source tokens found for the 
      greedy rationale of the `t + 1`-st target token. This is a list of length
      `tgt_len - 1` rather than `tgt_len` since we don't rationalize the first
      target token (this is usually a special token like <bos>).
    all_target_rationales: A list of length `tgt_len - 1` containing the target 
      tokens that form the best rationales for the corresponding target tokens. 
      The `t`-th entry of the list is a list of target tokens found for the 
      greedy rationale of the `t + 1`-st target token.
    log: A dictionary containing logging information from the search process.
  """
  all_source_rationales = []
  all_target_rationales = []
  log = {}

  num_source_tokens = len(source_tokens)
  num_target_tokens = len(target_tokens)

  source_tokens_text = decode_sequence(model, source_tokens, source=True)
  target_tokens_text = decode_sequence(model, target_tokens, source=False)
  log['source_tokens'] = list(source_tokens.cpu().numpy())
  log['target_tokens'] = list(target_tokens.cpu().numpy())
  log['source_tokens_text'] = source_tokens_text
  log['target_tokens_text'] = target_tokens_text
  log['rationalizations'] = []

  if verbose:
    print('All source tokens: {}'.format(source_tokens_text))
    print('All target tokens: {}'.format(target_tokens_text))
  
  # Get position IDs for all tokens.
  all_source_positions = utils.make_positions(
    source_tokens[None], model.model.encoder.padding_idx)
  all_target_positions = utils.make_positions(
    target_tokens[None], model.model.decoder.padding_idx)

  # Perform greedy rationalization for each token in the target sequence.
  # At step `target_token`, we are using the tokens `[0, target_token]` to 
  # predict token `target_token + 1`.
  for target_token in range(num_target_tokens - 1):
    goal_word_text = target_tokens_text[target_token + 1]
    token_log = {}
    token_log['target_position'] = target_token + 1
    token_log['goal_word'] = goal_word_text
    token_log['log'] = []

    # Initialize the rationales. The target rationale must always include the
    # most recent token.
    source_rationale = []
    target_rationale = [target_token]

    if verbose:
      print("Currently rationalizing token {}: '{}'".format(
        target_token + 1, 
        goal_word_text))
    
    max_possible_steps = target_token + num_source_tokens + 2
    max_rationale_size = min(max_steps + 1, max_possible_steps)
    for rationale_size in range(1, max_rationale_size):
      if rationale_size == 1:
        # A rationale of size 1 can only include the most recent target token.
        # Since there can be no source token input, we pass a dummy tensor to 
        # the encoder.
        encoder_out = model.model.encoder(
          torch.tensor([[model.model.encoder.padding_idx]]).to(model.device))
        # Our decoder input is the most recent target token.
        decoder_out = model.model.decoder(
          target_tokens[target_token:(target_token + 1)][None],
          encoder_out=encoder_out,
          position_ids=all_target_positions[:, target_token:(target_token + 1)]
        )
        # Since there's only one possible rationale of size 1, the most recent
        # token is by default the best for predicting the next token.
        best_probs = model.model.get_normalized_probs(
          decoder_out, log_probs=False)[0, -1]
        added_token_text = target_tokens_text[target_token]
        added_token_position = target_token
        added_token_from = "target"
        if verbose:
          added_token_string = ("Adding previous target token to sequence: "
                                "'{}'".format(added_token_text))
      else:
        # If all the target tokens are already in our rationale, we must 
        # rationalize the source tokens, and vice-versa.
        only_rationalize_source = len(target_rationale) == target_token + 1
        only_rationalize_target = len(source_rationale) == num_source_tokens
        # source_candidates contains the positions of each candidate rationale
        # from the source side
        source_candidates = [
          [x] + source_rationale for x in range(num_source_tokens) 
          if x not in source_rationale]
        
        if not only_rationalize_target:
          # If we are considering source tokens, create a batch that contains
          # each source candidate.
          encoder_src_tokens = source_tokens[[source_candidates]]
          encoder_positions = all_source_positions[0, source_candidates]
        
        if not only_rationalize_source:
          # If we are considering target tokens, create a batch of the current
          # source rationale without considering any other source candidates.
          num_target_candidates = (target_token + 1) - len(target_rationale)
          encoder_tokens_padded = source_tokens[[[
            source_rationale for _ in range(num_target_candidates)]]]
          if not only_rationalize_target:
            # If we are considering *both* source and target candidates, add
            # padding tokens to the end of the source batch to account for
            # the target-side candidates.
            padding_tokens = torch.ones(num_target_candidates, 1).to(
              encoder_tokens_padded) * model.model.encoder.padding_idx
            encoder_tokens_padded = torch.cat(
              [padding_tokens, encoder_tokens_padded], 1)
            # Get the position IDs of the pad indices
            pad_positions = [
              [0] + source_rationale for _ in range(num_target_candidates)]
          else:
            # If we are considering target tokens but not encoder tokens, we
            # don't need to add padding on the source side since all the 
            # batch elements have the same length. Thus the positions for the
            # padded indices are just the existing rationale.
            pad_positions = [
              source_rationale for _ in range(num_target_candidates)]
          encoder_positions_padded = all_source_positions[
            0, pad_positions]
        
        if only_rationalize_target:
          # If we are only considering candidates on the target side, the 
          # encoder inputs are just the padded batch entries.
          encoder_src_tokens = encoder_tokens_padded
          encoder_positions = encoder_positions_padded
        elif only_rationalize_source:
          # If we're not considering any target candidates, we don't need to
          # add any padding.
          pass
        else:
          # If we are considering both source and target candidates, add
          # padding to the end of the source batch.
          # Thus, the beginning of our batch will contain the source rationale
          # with each possible source token candidate, and the end of the batch
          # will contain only the current source rationale with the padding.
          encoder_src_tokens = torch.cat(
            [encoder_src_tokens, encoder_tokens_padded], 0)
          encoder_positions = torch.cat(
            [encoder_positions, encoder_positions_padded], 0)
        
        encoder_out = model.model.encoder(encoder_src_tokens, 
                                          position_ids=encoder_positions)
        num_current_target_tokens = target_token + 1

        if not only_rationalize_source:
          # If we're not only rationalizing from the source, consider target 
          # tokens for the rationale, using `sorted` to make sure the 
          # transformer causal masking is being applied correctly.
          target_candidates = [
            sorted(target_rationale + [x]) 
            for x in range(num_current_target_tokens) 
            if x not in target_rationale]
          decoder_input = target_tokens[[target_candidates]]
          decoder_positions = all_target_positions[0, target_candidates]
        
        if not only_rationalize_target:
          # If we're considering source tokens in the rationale, create a 
          # batch on the target side fixed at the current target rationale.
          num_source_candidates = num_source_tokens - len(source_rationale)
          current_target_rationale = target_tokens[[[
            sorted(target_rationale) for _ in range(num_source_candidates)]]]
          # Since we may also be considering target rationale candidates in the
          # same batch, we must batch so each token has the same length.
          target_padding = torch.ones(num_source_candidates, 1).to(
            current_target_rationale) * model.model.decoder.padding_idx
          decoder_input_padded = torch.cat(
            [target_padding, current_target_rationale], 1)
          # Get corresponding indices of the original sequence to be used for
          # position IDs, adding [0] to the beginning to account for padding.
          original_indices = [sorted([0] + target_rationale) 
                              for _ in range(num_source_candidates)]
          decoder_positions_padded = all_target_positions[0, original_indices]
        
        if only_rationalize_target:
          # If we're not considering any source candidates, we don't need to
          # add padding to the decoder input.
          pass
        elif only_rationalize_source:
          # If we're not considering any target candidates, our entire input
          # is the batch we've already padded, so we don't need to concatenate.
          decoder_input = decoder_input_padded
          decoder_positions = decoder_positions_padded
        else:
          # If we're considering both source and target candidates, concatenate
          # the candidate and padding tensors to create the full batch.
          decoder_input = torch.cat([decoder_input_padded, decoder_input], 0)
          decoder_positions = torch.cat(
            [decoder_positions_padded, decoder_positions], 0)

        decoder_out = model.model.decoder(decoder_input, 
                                          encoder_out=encoder_out, 
                                          position_ids=decoder_positions)
        probs = model.model.get_normalized_probs(
          decoder_out, log_probs=False)[:, -1]
        true_target_prob = probs[:, target_tokens[target_token + 1]]
        best_token = true_target_prob.argmax()
        best_probs = probs[best_token]

        # Add best token to rationale based on whether it's from source or
        # target
        if only_rationalize_source or (
          not only_rationalize_target and best_token < len(source_candidates)):
          # Best token came from source.
          best_token_position = set(
            source_candidates[best_token]) - set(source_rationale)
          best_token_position = best_token_position.pop()
          source_rationale.append(best_token_position)
          added_token = source_tokens_text[best_token_position]
          added_token_string = "Adding source token: '{}'".format(added_token)
          added_token_text = source_tokens_text[best_token_position]
          added_token_position = best_token_position
          added_token_from = "source"
        else:
          # Best token came from target.
          if only_rationalize_target:
            best_token_position = set(
              target_candidates[best_token]) - set(target_rationale)
          else:
            best_token_position = set(
              target_candidates[best_token - len(source_candidates)]) - set(
              target_rationale)
          best_token_position = best_token_position.pop()
          target_rationale.append(best_token_position)
          added_token = target_tokens_text[best_token_position]
          added_token_string = "Adding target token: '{}'".format(
            added_token)
          added_token_text = target_tokens_text[best_token_position]
          added_token_position = best_token_position
          added_token_from = "target"

      predicted_word_id = best_probs.argmax().item()
      predicted_word_prob = best_probs.max().item()
      predicted_word_text = decode_single_word(model, predicted_word_id, 
                                               source=False)
      true_token_prob = best_probs[target_tokens[target_token + 1]].item()
      
      token_log['log'].append({
        "rationale_size": rationale_size,
        "added_token_position": added_token_position,
        "added_token_text": added_token_text,
        "from": added_token_from,
        "prediction": predicted_word_text,
        "prediction_prob": predicted_word_prob,
        "true_token_prob": true_token_prob,
      })

      if verbose:
        print("{}. This makes the top predicted word: '{}'. "
              "P('{}') = {:.3f}".format(
                added_token_string, predicted_word_text, 
                goal_word_text, true_token_prob))
      # Search is done if the predicted token is the goal token, or if we're
      # doing top-1 rationalization and we've added a source token. We finish
      # if we've added 2 source tokens rather than 1 in case the first added
      # token is <eos>.
      if (
        (predicted_word_id == target_tokens[target_token + 1] and not top_1) or
        (top_1 and len(source_rationale) >= 2)):
        if verbose:
          print("When predicting: '{}'".format(goal_word_text))
          print("  The source rationale is: {}".format(
            ', '.join([source_tokens_text[x] for x in source_rationale])))
          print("  The target rationale is: {}".format(
            ', '.join([target_tokens_text[x] for x in target_rationale])))
          print("Finished with {} tokens: {} in source and "
                "{} in target.".format(rationale_size, 
                                       len(source_rationale), 
                                       len(target_rationale)))
          print("..........")
        break
    # When we're done rationalizing each target word, add the rationales to 
    # each rationale list and complete logging.
    all_source_rationales.append(source_rationale)
    all_target_rationales.append(target_rationale)
    token_log['source_rationale'] = source_rationale
    token_log['target_rationale'] = target_rationale
    reached_argmax = predicted_word_id == target_tokens[target_token + 1]
    token_log['reached_argmax'] = reached_argmax.item()
    log['rationalizations'].append(token_log)
  
  log['all_source_rationales'] = all_source_rationales
  log['all_target_rationales'] = all_target_rationales
  return all_source_rationales, all_target_rationales, log


@torch.no_grad()
def rationalize_lm(model,
                   input_ids,
                   verbose=False,
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
    model: A Fairseq model. It should be a `GeneratorHubInterface` object, with
      `model.model` as an object that implements the 
      `FairseqEncoderDecoderModel` interface. The model should be trained or
      fine-tuned for compatibility (i.e. with word dropout) to produce
      sensible rationales.
    input_ids: A tensor of shape `[seq_len]` containing the input IDs of the
      sequence to rationalize.
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

  input_text = decode_sequence(model, input_ids, source=False)
  log['input_ids'] = list(input_ids.cpu().numpy())
  log['input_text'] = input_text
  log['rationalization'] = []

  if verbose:
    print("All tokens: {}".format(input_text))
  
  all_positions = utils.make_positions(
    input_ids[None], model.model.decoder.padding_idx)
  
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
        decoder_out = model.model.decoder(
          input_ids[prev_token:(prev_token + 1)][None],
          position_ids=all_positions[:, prev_token:(prev_token + 1)])
        best_probs = model.model.get_normalized_probs(
          decoder_out, log_probs=False)[0, -1]
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
        candidate_position_ids = all_positions[0, candidates]

        # Divide the candidates into batches, since all possible subsets may
        # not fit in memory if we pass them to the model at once.
        num_candidates, seq_len = candidate_input_ids.shape
        batch_size = math.floor(max_tokens_per_batch / seq_len)
        num_batches = math.ceil(num_candidates / batch_size)
        best_prob = -float("inf")
        for batch_ind in range(num_batches):
          batch_start_ind = batch_ind * batch_size
          batch_end_ind = (batch_ind + 1) * batch_size
          batch_input_ids = candidate_input_ids[batch_start_ind:batch_end_ind]
          batch_position_ids = candidate_position_ids[
            batch_start_ind:batch_end_ind]
          batch_decoder_out = model.model.decoder(
            batch_input_ids,
            position_ids=batch_position_ids)
          # Only consider the probs of predicting the next token.
          # print(batch_input_ids.shape)
          # print(batch_position_ids.shape)
          # print(batch_decoder_out[0].shape)
          # print((model.model.get_normalized_probs(
          #   batch_decoder_out, log_probs=False)).shape)
          # print(input_ids[prev_token + 1])
          batch_probs = model.model.get_normalized_probs(
            batch_decoder_out, log_probs=False)[:, -1]
          true_token_probs = batch_probs[:, input_ids[prev_token + 1]]
          if batch_probs.max() > best_prob:
            best_prob = true_token_probs.max()
            best_token = true_token_probs.argmax() + batch_start_ind
            best_probs = batch_probs[true_token_probs.argmax()]
        
        best_token_position = set(candidates[best_token]) - set(rationale)
        best_token_position = best_token_position.pop()
        rationale.append(best_token_position)
        added_token = input_text[best_token_position]
        added_token_string = "Adding token: '{}'".format(added_token)
        added_token_text = input_text[best_token_position]
        added_token_position = best_token_position
      
      predicted_word_id = best_probs.argmax().item()
      predicted_word_prob = best_probs.max().item()
      predicted_word_text = decode_single_word(model, predicted_word_id, 
                                               source=False)
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
      if torch.argmax(best_probs) == input_ids[prev_token + 1]:
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
  return all_rationales, log
