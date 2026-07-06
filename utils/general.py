import transformer_lens
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt, ICLDataset
import torch
import random
import math
import os
import lrp_for_tl
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats

############################################################################################
# compute first order attributions for effect estimates

def get_cache(model, tokens, hook_names, out_indices_true, backward = True):
    
    cache = {}
    grad_cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()[:, -1]
    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()[:, -1]
    
    model.reset_hooks()
    for hook in hook_names:
        model.add_hook(hook, forward_cache_hook, "fwd")
        model.add_hook(hook, backward_cache_hook, "bwd")

    value = true_logits(forward_batched(model, tokens), out_indices_true)

    if backward:
        value.backward()

    model.reset_hooks()

    return (
        value.item(),
        cache,
        grad_cache
    )

def true_logits(logits, out_indices_true):
    return logits[:, -1, :].gather(1, out_indices_true.unsqueeze(1)).sum()

def run_attributions(
        model, hook_names, data, n_samples, batch_size, n_shots, backward = True):
    act = []
    grad = []
    samples = n_shot_indices(data, n_samples, n_shots)
    n_batches = math.ceil(n_samples / batch_size)
    for b in range(n_batches):
        s = samples[b*batch_size:(b+1)*batch_size]
        tokens, out_indices_true, _ = generate_batch(s, data, model)
        _, act_cache, grad_cache = get_cache(
            model, tokens, hook_names, out_indices_true, backward)
        act.append(torch.stack([act_cache[h] for h in hook_names]))
        if backward:
            grad.append(torch.stack([grad_cache[h] for h in hook_names]))
    act = torch.cat(act, dim=1)
    if backward:
        grad = torch.cat(grad, dim=1)
    return act, grad

############################################################################################
# data

def first_token_withspace(model, word):
    return model.to_tokens(" " + word, prepend_bos=False)[0][0].item()

def no_space_tokens(model, dataset):
    id_space = model.to_tokens(" ", prepend_bos=False).item()
    mask = dataset.raw_data.output.apply(
        lambda word: first_token_withspace(model, word) != id_space
    )
    dataset.raw_data = dataset.raw_data[mask].reset_index(drop=True)
    return dataset

def load_task_split(task, split, data_dir="data"):
    return ICLDataset(os.path.join(data_dir, split, f"{task}.json"))

# generate n_samples distinct n_shot samples- all test examples are distinct, and the ICL 
# examples are distinct from the test example
def n_shot_indices(data, n_samples, n_shots):
    test_indices = random.sample(range(len(data)), n_samples)
    indices = [
        random.sample([j for j in range(len(data)) if j != idx], n_shots) + [idx]
        for idx in test_indices
    ]
    return indices

############################################################################################
# data functions

def generate_batch(samples, dataset, model, padding_side = "left"):
    prompts = []
    test_pairs = []
    input_inds = []
    for sample in samples:
        input_pairs = dataset[sample[:-1]]
        test_pair = dataset[sample[-1]]
        input_inds.append(
            [model.to_tokens(item, prepend_bos = False)[0, 0].item() for item in (" " + test_pair["input"]).split(",")])
        prompt_data = word_pairs_to_prompt_data(
            input_pairs, 
            query_target_pair=test_pair, 
        )
        sentence = create_prompt(prompt_data)
        prompts.append(sentence)
        test_pairs.append([" "+test_pair["input"], " "+test_pair["output"]])
    tokens = model.tokenizer(
        prompts, return_tensors="pt", padding = True, padding_side = padding_side).to(
            model.cfg.device)
    out_indices_true = torch.tensor(
        [model.to_tokens(test_pairs[i][1], prepend_bos=False)[0][0] for i in range(len(test_pairs))],
        device=model.cfg.device,)
    return tokens, out_indices_true, input_inds

############################################################################################
# forward functions (including interventions)

def forward_batched(model, tokens):
    return model(input = tokens["input_ids"], attention_mask = tokens["attention_mask"])

def intervention_hooks(model, mask_dict, patch_dict, scale, hook_names):
    def intervention_hook(act, hook):
        mask = mask_dict[hook.name]
        patch = patch_dict[hook.name]
        act[:, -1] += scale*mask*patch
        return act
    model.reset_hooks()
    for hook in hook_names:
        model.add_hook(hook, intervention_hook, "fwd")
    return None

def intervention_fwd(model, tokens, mask_dict, patch_dict, scale, hook_names):
    intervention_hooks(model, mask_dict, patch_dict, scale, hook_names)
    out = forward_batched(model, tokens)
    model.reset_hooks()
    return out

############################################################################################
# mask functions

def topk_mask(model, hook_names, n_neurons, effects):
    _, top_indices = effects.flatten().topk(n_neurons)
    mask = torch.zeros_like(effects).flatten()
    mask[top_indices] = 1
    mask = mask.view_as(effects)
    mask_dict = {hook_names[i]: mask[i] for i in range(model.cfg.n_layers)}
    return mask_dict, mask

def topk_mask_bool(n_neurons, effects):
    _, top_indices = effects.flatten().topk(n_neurons)
    mask = torch.zeros_like(effects,  dtype=torch.bool).flatten()
    mask[top_indices] = True
    mask = mask.view_as(effects)
    return mask

############################################################################################
# evaluation functions

def evaluate_accuracy(model, fwd_fun, data, n_samples, n_shots, batch_size, samples=None,
                      **kwargs):
    if samples is None:
        samples = n_shot_indices(data, n_samples, n_shots)
    n_batches = math.ceil(n_samples / batch_size)
    n_corr = 0
    for b in range(n_batches):
        s = samples[b*batch_size:(b+1)*batch_size]
        tokens, out_indices_true, _ = generate_batch(s, data, model)
        pred = fwd_fun(model, tokens, **kwargs)
        pred_indices = pred.softmax(dim=-1)[:, -1, :].argmax(dim=-1)
        n_corr += (out_indices_true == pred_indices).sum().item()
    accuracy = n_corr / n_samples
    return accuracy

def eval_hyperpars(model, fwd_fun, conditions, hook_names, effects, dataset,
                   n_samples, batch_size, mask_dicts=None, **kwargs):
    samples = n_shot_indices(dataset, n_samples, 0)
    accuracies = []
    for c in tqdm(conditions):
        scale = c[0]
        n_neurons = c[1]
        if mask_dicts is None:
            mask_dict, _ = topk_mask(model, hook_names, n_neurons, effects)
        else:
            mask_dict = mask_dicts[n_neurons]
        acc = evaluate_accuracy(
            model, fwd_fun, dataset, n_samples, 0, batch_size,
            samples=samples,
            mask_dict=mask_dict,
            scale=scale,
            hook_names=hook_names,
            **kwargs)
        accuracies.append(acc)
    return accuracies

############################################################################################
# helpers

def write_taskwise(df, results_dir, taskwise_sub, model_short, task):
    """Write a single (model, task) result shard so parallel jobs don't collide."""
    d = os.path.join(results_dir, taskwise_sub)
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, "_".join([model_short, task]) + ".csv"), index=False)

def combine_taskwise(model_name, tasks):
    """Merge per-(model, task) shards into the per-model CSVs that figures.R reads."""
    m = model_name.split("/")[-1]
    jobs = [
        ("results/baseline",  "taskwise",            ""),
        ("results/iterative", "taskwise",            ""),
        ("results/iterative", "taskwise_summary",    "_summary"),
        ("results/test",      "taskwise_robust",     "_robust"),
        ("results/test",      "taskwise_not_robust", "_not_robust"),
        ("results/hyperpars", "taskwise_robust",     "_robust"),
        ("results/hyperpars", "taskwise_not_robust", "_not_robust"),
    ]
    for results_dir, taskwise_sub, suffix in jobs:
        src = os.path.join(results_dir, taskwise_sub)
        if not os.path.isdir(src):
            continue
        parts = [
            pd.read_csv(os.path.join(src, "_".join([m, t]) + ".csv"))
            for t in tasks
            if os.path.exists(os.path.join(src, "_".join([m, t]) + ".csv"))
        ]
        if parts:
            pd.concat(parts).to_csv(
                os.path.join(results_dir, m + suffix + ".csv"), index=False)

def load_attributions(
        model_name, task, grad_type,
        map_location=torch.device("cuda"),
        folder = "attributions"):
    return torch.load(
        folder + "/" + "_".join([model_name, task, grad_type]) + ".pt",
        map_location=map_location)

def intervention_effect(attributions):
    act_diff = (attributions["act_n"] - attributions["act_0"]).mean(dim=1)
    gradient = attributions["grad_0"].mean(dim=1)
    effect = gradient*act_diff
    return act_diff, effect

def wilson_ci(p, n, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z/(2*n) * np.sqrt(4*n*p*(1 - p) + z**2) / denom
    return center - half, center + half    

def load_model(model_name):
    model = transformer_lens.HookedTransformer.from_pretrained(model_name)
    model.eval()
    hook_names = [transformer_lens.utils.get_act_name("z", i, "a") for i in range(model.cfg.n_layers)]
    return model, hook_names

