import os
import pandas as pd
import torch
import itertools
import random
import math
import json
from utils.general import no_space_tokens, load_task_split, forward_batched, \
    first_token_withspace, evaluate_accuracy, load_model, load_attributions, \
    intervention_effect, topk_mask_bool, n_shot_indices, generate_batch, intervention_fwd, \
    wilson_ci, eval_hyperpars, run_attributions, write_taskwise, combine_taskwise
from utils.iterative import prox_l0_ball, scale_grad, scale_value
from ZeroFPR import ZeroFPR
from numpy import linspace

def eval_baseline(
        model_name,
        tasks,
        n_shots,
        n_samples,
        forward_batch_size_0_shot,
        forward_batch_size_n_shot,
        torch_seed,
        random_seed):

    model, _ = load_model(model_name)

    for task in tasks:

        torch.manual_seed(torch_seed)
        random.seed(random_seed)

        ####################################################################################
        # load data

        data = no_space_tokens(model, load_task_split(task, "test"))
        n_samples_task = min(n_samples, len(data))

        ####################################################################################
        # n-shot baselines

        accuracy_0_shot = evaluate_accuracy(
            model,
            forward_batched,
            data,
            n_samples_task,
            0,
            forward_batch_size_0_shot)

        accuracy_n_shot = evaluate_accuracy(
            model,
            forward_batched,
            data,
            n_samples_task,
            n_shots,
            forward_batch_size_n_shot)

        ####################################################################################
        # majority class baseline ("always predict the most common output token")

        output_tokens = \
            torch.tensor([first_token_withspace(model, word) for word in data.raw_data.output])
        most_common_token = output_tokens.mode().values.item()
        accuracy_majority = (output_tokens == most_common_token).float().mean().item()

        row = {
            "task": task,
            "n_shots" : n_shots,
            "nsamples": n_samples_task,
            "0_shot_accuracy": accuracy_0_shot,
            "n_shot_accuracy": accuracy_n_shot,
            "majority_accuracy": accuracy_majority,
        }

        write_taskwise(
            pd.DataFrame([row]), "results/baseline", "taskwise",
            model.cfg.model_name, task)

def compute_attributions(
        model_name,
        tasks,
        n_shots,
        n_samples,
        forward_batch_size_0_shot,
        forward_batch_size_n_shot,
        backward_batch_size,
        torch_seed,
        random_seed,
        robust):

    os.makedirs("attributions", exist_ok=True)

    model, hook_names = load_model(model_name) 

    # patch model
    if robust:
        model.patch_lrp()
        model.lrp_on()
        model.set_lrp_rules(
            'half-rule-attn','half-rule-mlp','identity-rule-act','ln-rule-pre')

    for task in tasks:

        torch.manual_seed(torch_seed)
        random.seed(random_seed)

        ####################################################################################
        # load data

        data = no_space_tokens(model, load_task_split(task, "train"))
        n_samples_task = min(n_samples, len(data))

        ####################################################################################
        # 0-shot activations and gradients

        act_0, grad_0 = run_attributions(
            model, hook_names, data, n_samples_task,
            min(forward_batch_size_0_shot, backward_batch_size),
            0, backward=True)

        ####################################################################################
        # n-shot activations only

        act_n, _ = run_attributions(
            model, hook_names, data, n_samples_task,
            forward_batch_size_n_shot,
            n_shots, backward=False)

        ####################################################################################
        # save to disk

        grad_type = "robust" if robust else "not_robust"

        torch.save(
            {"act_0" : act_0, "act_n" : act_n, "grad_0" : grad_0},
            "attributions/" + "_".join([model.cfg.model_name, task, grad_type]) + ".pt")
        

def compute_iterative(
        model_name,
        tasks,
        sizes,
        n_samples,
        batch_size,
        max_iterations,
        tol,
        torch_seed,
        random_seed,
        n_restarts=10,
        noise_scale=0.3,
        ):
    
    os.makedirs("iterative/interventions", exist_ok=True)
    os.makedirs("iterative/trace", exist_ok=True)

    ############################################################################################
    # load model

    model, hook_names = load_model(model_name)
    model.requires_grad_(False)

    for task in tasks:

        torch.manual_seed(torch_seed)
        random.seed(random_seed)

        summary_rows = []

        data = no_space_tokens(model, load_task_split(task, "train"))
        n_samples_task = min(n_samples, len(data))

        samples = n_shot_indices(data, n_samples_task, 0)

        attributions = load_attributions(model.cfg.model_name, task, "robust")
        act_diff, effect = intervention_effect(attributions)

        n_batches = math.ceil(n_samples_task / batch_size)
        batches = []
        for b in range(n_batches):
            sub_samples = samples[b*batch_size:(b+1)*batch_size]
            tokens, out_indices_true, _ = generate_batch(sub_samples, data, model)
            batches.append((tokens, out_indices_true))

        for n_neurons in sizes:

            # penalty function
            l0 = lambda y, gamma, k=n_neurons: prox_l0_ball(y, k)
            # inital values
            mask = topk_mask_bool(n_neurons, effect)
            s_0 = torch.zeros_like(act_diff)
            s_0[mask] = 1.0

            # objective function
            f = lambda s: scale_value(s, act_diff, model, hook_names, batches, n_samples_task)
            f_and_grad = lambda s: scale_grad(s, act_diff, model, hook_names, batches, n_samples_task)
            initial_value, _ = f_and_grad(s_0)

            best_s, best_trace, best_final_value = None, None, float("-inf")

            for restart in range(n_restarts):
                if restart == 0:
                    x0 = s_0.clone()
                else:
                    x0 = noise_scale * torch.randn_like(s_0)

                s, trace_r = ZeroFPR(
                    x0=x0,
                    f=f,
                    f_and_grad=f_and_grad,
                    prox_g=l0,
                    max_iter=max_iterations,
                    tol=tol,
                )
                # iterative interventions
                interventions_r = s * act_diff
                patch_dict = {hook_names[l]: interventions_r[l] for l in range(model.cfg.n_layers)}
                mask_dict = {hook_names[l]: torch.ones_like(interventions_r[l], dtype=torch.bool)
                             for l in range(model.cfg.n_layers)}
                # training set performance
                n_corr = 0
                for tokens, out_indices_true in batches:
                    pred = intervention_fwd(model, tokens, mask_dict=mask_dict, patch_dict=patch_dict,
                                    scale=1, hook_names=hook_names)
                    pred_indices = pred.softmax(dim=-1)[:, -1, :].argmax(dim=-1)
                    n_corr += (out_indices_true == pred_indices).sum().item()
                train_accuracy = n_corr / n_samples_task
                # keep best run
                if train_accuracy > best_final_value:
                    best_s, best_trace, best_final_value = s, trace_r, train_accuracy
                    best_iter = restart

            s, trace, final_value = best_s, best_trace, best_final_value

            # save interventions
            interventions = s * act_diff
            path = os.path.join(
                "iterative",
                "interventions",
                "_".join([model.cfg.model_name, task, str(n_neurons)]) + ".pt"
            )
            torch.save(interventions, path)

            trace = [-initial_value] + [-v for v in trace]
            mask_overlap = ((interventions != 0) & (mask == 1)).sum().item()

            # save trace
            trace_path = os.path.join(
                "iterative",
                "trace",
                "_".join([model.cfg.model_name, task, str(n_neurons)]) + ".json"
            )
            with open(trace_path, "w") as f:
                json.dump(
                    {"trace": trace, "best_iter": best_iter}, f)

            summary_rows.append({
                "task": task,
                "size": n_neurons,
                "mask_overlap": mask_overlap,
                "train_accuracy": final_value,
                "n_iter": len(trace) - 1,
            })

        write_taskwise(
            pd.DataFrame(summary_rows), "results/iterative", "taskwise_summary",
            model.cfg.model_name, task)

def eval_iterative(
        model_name, 
        tasks, 
        n_samples, 
        batch_size, 
        sizes, 
        alpha, 
        torch_seed, 
        random_seed):
    
    model, hook_names = load_model(model_name)

    for task in tasks:

        torch.manual_seed(torch_seed)
        random.seed(random_seed)

        rows = []

        data = no_space_tokens(model, load_task_split(task, "test"))
        n_samples_task = min(len(data), n_samples)

        ####################################################################################
        # evaluate on test set

        for n_neurons in sizes:
            # load interventions
            path = os.path.join(
                "iterative",
                "interventions",
                "_".join([model.cfg.model_name, task, str(n_neurons)]) + ".pt"
            )
            interventions = torch.load(path)
            patch_dict = {hook_names[l]: interventions[l] for l in range(model.cfg.n_layers)}
            mask_dict = {hook_names[l]: torch.ones_like(interventions[l], dtype=torch.bool)
                         for l in range(model.cfg.n_layers)}
            # compute accuracy
            accuracy = evaluate_accuracy(
                model, intervention_fwd, data, n_samples_task, 0, batch_size,
                mask_dict=mask_dict, patch_dict=patch_dict, scale=1, hook_names=hook_names)
            ci_lower, ci_upper = wilson_ci(accuracy, n_samples_task, alpha)
            rows.append({
                "size": n_neurons, "size_check": (interventions != 0).sum().item(),
                "accuracy": accuracy, "ci_lower": ci_lower, "ci_upper": ci_upper,
                "n_samples": n_samples_task, "task": task})

        write_taskwise(
            pd.DataFrame(rows), "results/iterative", "taskwise",
            model.cfg.model_name, task)

def eval_first_order(
        model_name, 
        tasks, 
        n_train,
        n_test, 
        batch_size, 
        sizes,
        scales,
        alpha, 
        torch_seed, 
        random_seed,
        robust):
    
    model, hook_names = load_model(model_name)

    grad_type = "robust" if robust else "not_robust"

    conditions = list(itertools.product(scales, sizes))

    for task in tasks:

        torch.manual_seed(torch_seed)
        random.seed(random_seed)

        train_data = no_space_tokens(model, load_task_split(task, "train"))
        test_data = no_space_tokens(model, load_task_split(task, "test"))

        n_train_task = min(len(train_data), n_train)
        n_test_task = min(len(test_data), n_test)

        ####################################################################################
        # load effect estimates

        attributions = load_attributions(model.cfg.model_name, task, grad_type)
        act_diff, effect = intervention_effect(attributions)
        patch_dict = {hook_names[l]: act_diff[l] for l in range(model.cfg.n_layers)}

        ####################################################################################
        # evaluate hyperpars on training set

        accuracies = eval_hyperpars(
            model, intervention_fwd, conditions, hook_names, effect, train_data, n_train_task, batch_size,
            patch_dict=patch_dict)

        hyperpar_res = pd.DataFrame(conditions, columns=['scale', 'size'])
        hyperpar_res["accuracy"] = accuracies
        hyperpar_res["task"] = task
        write_taskwise(
            hyperpar_res, "results/hyperpars", "taskwise_" + grad_type,
            model.cfg.model_name, task)

        ####################################################################################
        # evaluate best hyperpars on test set

        best_scales = [
            hyperpar_res[hyperpar_res["size"] == s].sort_values("accuracy", ascending=False).scale.iloc[0]
            for s in sizes
        ]

        accuracies = eval_hyperpars(
            model, intervention_fwd, list(zip(best_scales, sizes)), hook_names, effect, test_data,
            n_test_task, batch_size, patch_dict=patch_dict)

        lowers, uppers = zip(*[wilson_ci(a, n_test_task, alpha) for a in accuracies])
        test_res = pd.DataFrame({
            "task": task,
            "accuracy": accuracies,
            "ci_lower": lowers,
            "ci_upper": uppers,
            "scale": best_scales,
            "size": sizes,
            "nsamples": n_test_task
        })
        write_taskwise(
            test_res, "results/test", "taskwise_" + grad_type,
            model.cfg.model_name, task)

def run_intervention_effects(
    torch_seed,
    random_seed,
    n_pairs = 6,
    n_neurons = 16,
    n_steps = 21,
    scale_max = 40,
    n_samples = 100,
    model_name = "meta-llama/Llama-3.2-3B-Instruct",
    task = "english-french"):

    os.makedirs("results/interactions", exist_ok=True)

    ########################################################################################
    # load model

    model, hook_names = load_model(model_name)

    ########################################################################################
    # loop over tasks

    all_rows = []

    torch.manual_seed(torch_seed)
    random.seed(random_seed)

    ####################################################################################
    # load data

    dataset = no_space_tokens(model, load_task_split(task, "test"))

    nsamples_test = min(len(dataset), n_samples)

    ############################################################################################
    # load attributions

    attributions = load_attributions(model.cfg.model_name, task, "robust")
    act_diff, effect = intervention_effect(attributions)

    # get flat indices of top-n_neurons by relevance
    top_indices = effect.flatten().topk(n_neurons).indices

    ############################################################################################
    # sample test data (fixed across all pairs for comparability)

    samples = random.sample(range(len(dataset)), nsamples_test)
    samples = [[s] for s in samples]
    tokens, out_indices_true, _ = generate_batch(samples, dataset, model)

    scales = linspace(0, scale_max, n_steps)

    ############################################################################################
    # sweep random pairs from the top-n_neurons pool

    pairs = [random.sample(range(n_neurons), 2) for _ in range(n_pairs)]
    r_flat = effect.flatten()

    for pair_idx, (a, b) in enumerate(pairs):

        mask1 = torch.zeros_like(r_flat)
        mask2 = torch.zeros_like(r_flat)
        mask1[top_indices[a]] = 1
        mask2[top_indices[b]] = 1
        mask1 = mask1.view_as(effect)
        mask2 = mask2.view_as(effect)

        for s1 in scales:
            for s2 in scales:
                combined = s1 * mask1 * act_diff + s2 * mask2 * act_diff
                union_mask = (mask1 + mask2).bool()

                combined_patch_dict = {hook_names[l]: combined[l] for l in range(model.cfg.n_layers)}
                union_mask_dict = {hook_names[l]: union_mask[l] for l in range(model.cfg.n_layers)}

                logits = intervention_fwd(
                    model, tokens,
                    mask_dict=union_mask_dict, patch_dict=combined_patch_dict,
                    scale=1, hook_names=hook_names)
                fval = logits[:, -1, :].gather(
                    1, out_indices_true.unsqueeze(1)).mean().item()

                all_rows.append({
                    "task": task,
                    "pair": pair_idx,
                    "s1": s1,
                    "s2": s2,
                    "fval": fval,
                    "ind_1": a,
                    "ind_2": b,
                })

    pd.DataFrame(all_rows).to_csv(os.path.join("results", "interactions", "interactions.csv"))

def prepare_figures(model_name, tasks, sizes):

    # merge per-(model, task) shards written by the parallel compute jobs into the
    # per-model CSVs that figures.R reads (and that this step itself consumes below)
    combine_taskwise(model_name, tasks)

    model_short = model_name.split("/")[-1]
    test_res = pd.read_csv(os.path.join("results", "test", model_short + "_robust.csv"))
    os.makedirs("results/figures/intervention_strength", exist_ok=True)
    os.makedirs("results/figures/neurons_per_head", exist_ok=True)

    rows = []
    neuron_rows = []
    for task in tasks:

        ####################################################################################
        # load attributions

        attributions = load_attributions(model_short, task, "robust")
        act_diff, effect = intervention_effect(attributions)

        task_res = test_res[test_res["task"] == task].set_index("size")

        for size in sizes:

            scale      = task_res.loc[size, "scale"]
            fo_mask    = topk_mask_bool(size, effect)
            total_abs  = (scale * act_diff[fo_mask]).abs().sum().item()
            rows.append({
                "task":      task,
                "method":    "first_order",
                "size":      size,
                "total_abs": total_abs,
                "scale_min": scale,
                "scale_max": scale,
            })

            fo_counts = fo_mask.sum(dim=-1)  # [n_layers, n_heads]
            for layer in range(fo_counts.shape[0]):
                for head in range(fo_counts.shape[1]):
                    neuron_rows.append({
                        "task":      task,
                        "method":    "first_order",
                        "size":      size,
                        "layer":     layer,
                        "head":      head,
                        "n_neurons": fo_counts[layer, head].item(),
                    })

            interventions = torch.load(os.path.join("iterative", "interventions",
                    "_".join([model_short, task, str(size)]) + ".pt"))
            it_mask       = interventions != 0

            total_abs = interventions[it_mask].abs().sum().item()

            rows.append({
                "task":      task,
                "method":    "iterative",
                "size":      size,
                "total_abs": total_abs
            })

            it_counts = it_mask.sum(dim=-1)  # [n_layers, n_heads]
            for layer in range(it_counts.shape[0]):
                for head in range(it_counts.shape[1]):
                    neuron_rows.append({
                        "task":      task,
                        "method":    "iterative",
                        "size":      size,
                        "layer":     layer,
                        "head":      head,
                        "n_neurons": it_counts[layer, head].item(),
                    })

    pd.DataFrame(rows).to_csv(
        os.path.join("results", "figures", "intervention_strength", model_short + ".csv"),
        index=False)
    pd.DataFrame(neuron_rows).to_csv(
        os.path.join("results", "figures", "neurons_per_head", model_short + ".csv"),
        index=False)