import functools
import itertools
import torch


def precompute_attention_masks(num_tokens, num_heads, block_query_ranges, block_key_ranges,
                               dtype, device, opposite=False):
    """
    Precomputes attention masks for a given set of query and key ranges.
    """

    # Initialize the attention mask
    attn_mask = torch.tril(torch.ones((num_tokens, num_tokens), dtype=torch.uint8, device=device))

    if opposite:
        attn_mask = 1 - attn_mask  # Invert for opposite mode

    for block_query_range, block_key_range in zip(block_query_ranges, block_key_ranges):
        # Convert inputs to tensors if they are lists
        block_query_range = torch.tensor(block_query_range, device=device)
        block_key_range = torch.tensor(block_key_range, device=device)

        if block_query_range.size(0) > 0 and block_key_range.size(0) > 0:
            # Create indices for the mask modification
            q_indices = block_query_range[:, None]  # Make it a column vector
            s_indices = block_key_range[None, :]  # Make it a row vector

            # Modify the mask using advanced indexing
            attn_mask[q_indices, s_indices] = 0 if not opposite else 1

    attn_mask = attn_mask.repeat(1, num_heads, 1, 1).to(dtype=dtype)
    attn_mask = (1.0 - attn_mask) * torch.finfo(dtype).min  # Convert to large negative values for masking

    return attn_mask


def precompute_random_block_attention_masks(num_tokens, num_heads, valid_indices, N, dtype, device, opposite=False):
    # Initialize mask
    attn_mask = torch.tril(torch.ones((num_tokens, num_tokens), dtype=torch.uint8, device=device))

    # Select random indices vectorized
    permuted_indices = valid_indices[torch.randperm(valid_indices.size(0), device=device)[:N]]

    # Assign values directly
    attn_mask[permuted_indices[:, 0], permuted_indices[:, 1]] = 0 if not opposite else 1

    # Expand for multi-head and dtype conversion efficiently
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(-1, num_heads, -1, -1).to(dtype=dtype)
    attn_mask = (1.0 - attn_mask) * torch.finfo(dtype).min

    return attn_mask


@torch.no_grad()
def trace_with_attn_block(model, inp, answers_t, gt_t, attn_mask, layerlist, model_type='llava'):
    """
    Runs model forward with blocked attention, applying precomputed masks only to layers in layerlist.
    """
    with torch.no_grad():
        block_attn_hooks = _set_block_attn_hooks(model, attn_mask, layerlist, model_type)
        outputs_exp = model(**inp)
        _remove_wrapper(model, block_attn_hooks, model_type)

    probs = torch.softmax(outputs_exp.logits[0, -1, :], dim=0)
    prob_answer = probs[answers_t]  # prob of baseline pred class
    prob_gt = probs[gt_t]  # prob of gt class
    prob_max, prob_max_id = torch.max(probs, dim=0)  # prob of current pred class
    return prob_answer, prob_gt, prob_max, prob_max_id


@torch.no_grad()
def generate_with_attn_block(model, processor, inputs, conv, split_tag,
                             answers_t, gt_t, attn_mask, layerlist, model_type='llava', max_new_tokens=100):
    """
    Runs model generation with blocked attention, applying precomputed masks only to layers in layerlist.
    """
    block_attn_hooks = _set_block_attn_hooks(model, attn_mask, layerlist, model_type)

    output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                            num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0,
                            length_penalty=1, temperature=1.0, output_scores=True, return_dict_in_generate=True)

    _remove_wrapper(model, block_attn_hooks, model_type)

    scores = output.scores[0]   # first token score of generated answer
    probs = torch.softmax(scores[0, :], dim=0)
    prob_answer = probs[answers_t]  # prob of baseline pred class
    prob_gt = probs[gt_t]  # prob of gt class
    prob_max, prob_max_id = torch.max(probs, dim=0)  # prob of current pred class

    # Decode into text
    output_text = processor.batch_decode(output.sequences, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]

    # Gather answers without prompt inputs
    output_text = output_text.split(split_tag)[-1]
    ending = conv.sep if isinstance(conv.sep, str) else conv.sep[1]
    output_text = output_text.removesuffix(ending).strip()

    return prob_answer, prob_gt, prob_max, prob_max_id, output_text


def _set_block_attn_hooks(model, attn_mask, layerlist, model_type):
    """
    Wraps attention forward functions with precomputed attention masks
    """

    def wrap_attn_forward(forward_fn, mask):
        @functools.wraps(forward_fn)
        def wrapper_fn(*args, **kwargs):
            query_len_cur = kwargs["position_ids"].size(-1)
            key_len_cur = kwargs["attention_mask"].size(-1)

            if query_len_cur == key_len_cur:    # First forward
                kwargs["attention_mask"] = mask
            else:                               # Autoregressive answer generation
                b, h, q, k = mask.shape
                new_mask = torch.zeros((b, h, 1, key_len_cur), dtype=mask.dtype, device=mask.device)
                new_mask[:, :, 0, :k] = mask[:, :, -1, :k]
                kwargs["attention_mask"] = new_mask
            return forward_fn(*args, **kwargs)

        return wrapper_fn

    if model_type == 'videollama':
        target_module = model.model
    elif model_type in ['llava', 'internvl']:
        target_module = model.language_model.model
    elif model_type in ['llava_lora', 'internvl_lora']:
        target_module =model.language_model.base_model.model.model
    else:
        raise NotImplementedError

    hooks = []
    if layerlist is None:   # Apply hooks only to relevant layers in `layerlist`
        num_layers = len(attn_mask)
        for layer in range(num_layers):
            hook = target_module.layers[layer].self_attn.forward
            target_module.layers[layer].self_attn.forward = wrap_attn_forward(hook, attn_mask[layer])
            hooks.append((layer, hook))
    else:
        for layer in layerlist:   # Use predefined masks in all layers
            hook = target_module.layers[layer].self_attn.forward
            target_module.layers[layer].self_attn.forward = wrap_attn_forward(hook, attn_mask)
            hooks.append((layer, hook))

    return hooks


def _remove_wrapper(model, hooks, model_type='llava'):
    """
    Remove wrapper
    """

    if model_type == 'videollama':
        target_module = model.model
    elif model_type in ['llava', 'internvl']:
        target_module = model.language_model.model
    elif model_type in ['llava_lora', 'internvl_lora']:
        target_module =model.language_model.base_model.model.model
    else:
        raise NotImplementedError

    for i, hook in hooks:
        target_module.layers[i].self_attn.forward = hook


@torch.no_grad()
def predict_from_input(model, inp):
    """
    Next token prediction (one token)
    """
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p, probs


@torch.no_grad()
def generate_from_input(model, processor, inputs, conv, split_tag, max_new_tokens=100, return_hidden_states=False):
    """
    Next token prediction (multiple tokens)
    """
    output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                            num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0,
                            length_penalty=1, temperature=1.0, output_scores=True, return_dict_in_generate=True,
                            output_hidden_states=return_hidden_states)

    scores = output.scores[0]   # first token score of generated answer
    probs = torch.softmax(scores, dim=1)
    p, preds = torch.max(probs, dim=1)

    # Decode into text
    output_text = processor.batch_decode(output.sequences, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]

    # Gather answers without prompt inputs
    output_text = output_text.split(split_tag)[-1]
    ending = conv.sep if isinstance(conv.sep, str) else conv.sep[1]
    output_text = output_text.removesuffix(ending).strip()

    if return_hidden_states:
        # output.hidden_states: tuple(gen_seq_len) of tuple(num_layer + 1) of tensor(batch_size, total_seq_len, dim)
        # num_layer + 1 -> input (=input id embeddings) and output hidden representations at each layer
        hidden_states = output.hidden_states[0][1:]  # all layers' hidden representations at first generation step
        return preds[0], p[0], probs[0], output_text, hidden_states
    else:
        return preds[0], p[0], probs[0], output_text


@torch.no_grad()
def logit_lens_trace_with_proj(model, inp):
    # set hooks
    hooks = _logit_lens_set_proj_hooks(model)

    # get prediction
    answer_t, base_score, probs = [d[0] for d in predict_from_input(model, inp)]

    # remove hooks
    _remove_hooks(hooks)

    projs = model.projs_
    # {'layer_residual_0_preds': ndarray (b, seq_len),
    #  'layer_residual_0_probs': ndarray (b, seq_len),
    #  'layer_residual_1_preds': ndarray (b, seq_len),
    #  ..., }

    return answer_t, base_score, projs, probs


def _logit_lens_set_proj_hooks(model):
    for attr in ["projs_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_projection(name, E):
        def hook(module, input, output):    # called after forward()
            if name == f"layer_residual_{final_layer}":
                hs = output     # For final layer, use embeddings after layer norm projection
            else:
                hs = input[0]   # Otherwise, use input embeddings of l-th layer (i.e, output of (l-1)-th layer)
            probs, preds = torch.topk(
                torch.softmax(hs.matmul(E.T), dim=-1),
                k=10,  # Replace with the desired k value
                dim=-1
            )
            model.projs_[f"{name}_preds"] = preds.cpu().numpy()
            model.projs_[f"{name}_probs"] = probs.cpu().float().numpy()

        return hook

    E = model.get_input_embeddings().weight.detach()
    # E = model.get_output_embeddings().weight.detach()
    final_layer = model.config.text_config.num_hidden_layers - 1

    hooks = []
    for i in range(model.config.text_config.num_hidden_layers - 1):   # => LlamaDecoderLayer
        hooks.append(model.language_model.model.layers[i].register_forward_hook(
            get_projection(f"layer_residual_{i}", E)
        ))
    hooks.append(model.language_model.model.norm.register_forward_hook(   # => norm after LlamaDecoderLayer
        get_projection(f"layer_residual_{final_layer}", E)
    ))

    return hooks


# Always remove your hooks, otherwise things will get messy.
def _remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring, remove_margin=True):
    """Find the tokens corresponding to the given substring in token_array."""
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    if remove_margin:
        substring = substring.replace(" ", "")

    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def reverse_check_token_range(tokenizer, input_ids, start_idx, end_idx, substring):
    """
    Convert token IDs in the range [start_idx, end_idx] back to a string and compare with the substring.

    Args:
    - tokenizer: The tokenizer used for encoding and decoding.
    - input_ids: The list of token IDs.
    - start_idx: The starting index of the token range.
    - end_idx: The ending index of the token range.
    - substring: The substring to compare with the decoded string.

    Returns:
    - True if the decoded string matches the substring, False otherwise.
    """
    # Extract token IDs within the range
    token_ids_in_range = input_ids[start_idx:end_idx]

    # Decode the token IDs back into a string
    decoded_string = tokenizer.decode(token_ids_in_range, skip_special_tokens=True)

    # Compare the decoded string with the substring
    return decoded_string == substring


def find_inter_frame_block_ranges(vision_range, num_frames, num_vis_one_frame, vis_start_id):
    range_sep_by_frames = [[] for _ in range(num_frames)]
    for x in vision_range:
        frame_idx = (x - vis_start_id) // num_vis_one_frame
        range_sep_by_frames[frame_idx].append(x)

    query_lists, key_lists = [], []
    for i in range(1, num_frames):
        query_lists.append(range_sep_by_frames[i])
        key_lists.append(list(itertools.chain.from_iterable(range_sep_by_frames[:i])))  # block previous ranges

    return query_lists, key_lists