import argparse
import gc
import torch
import random
from steps import eval_baseline, compute_attributions, compute_iterative, eval_iterative, \
    eval_first_order, run_intervention_effects, prepare_figures
from numpy import logspace, linspace, concatenate

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch_seed = 21879
random_seed = 34782
torch.manual_seed(torch_seed)
random.seed(random_seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=str, nargs="+",
        default=[
            "baseline",
            "attributions",
            "compute_iterative",
            "eval_iterative",
            "no_robustification",
            "no_refinement",
            "prepare_figures",
            "intervention_effects"])
    parser.add_argument(
        "--model-names", type=str, nargs="+",
        default=[
            "Qwen/Qwen3-8B",
            "google/gemma-3-4b-it",
            "meta-llama/Llama-3.2-3B-Instruct"])
    parser.add_argument(
        "--tasks", type=str, nargs="+",
        default=[
            "capitalize",
            "country-capital", 
            "english-french",
            "present-past",
            "sentiment",
            "product-company",
            "fruit_v_animal_3",
            "color_v_animal_3",
            "object_v_concept_3",
            "choose_first_of_3",
            "adjective_v_verb_3",
            "choose_middle_of_3"
        ])
    # hyperpars
    parser.add_argument("--sizes", type=int, nargs="*", default=None)
    parser.add_argument("--n-shots", type=int, default=10)
    parser.add_argument("--scales", type=float, nargs="*", default=None)
    # sample sizes
    parser.add_argument("--n-baseline", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=200)
    # batch sizes
    parser.add_argument("--forward-batch-size-0-shot", type=int, default=50)
    parser.add_argument("--forward-batch-size-n-shot", type=int, default=1)
    parser.add_argument("--backward-batch-size", type=int, default=50)
    ##
    parser.add_argument("--alpha", type=float, default=0.05)
    # iterative
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--noise-scale", type=float, default=0.3)

    args = parser.parse_args()
    steps = args.steps
    n_shots = args.n_shots
    tasks = args.tasks
    n_train = args.n_train
    n_test = args.n_test
    sizes = args.sizes
    scales = args.scales
    # batch sizes
    forward_batch_size_0_shot = args.forward_batch_size_0_shot
    forward_batch_size_n_shot = args.forward_batch_size_n_shot
    backward_batch_size = args.backward_batch_size

    if sizes is None:
        sizes = [int(s) for s in logspace(3, 11, num=9, base=2)]
    if scales is None:
        scales = concatenate([linspace(0.01, 1, 10), linspace(2, 10, 9), linspace(11, 50, 30)])

    def flush():
        gc.collect()
        torch.cuda.empty_cache()

    for model_name in args.model_names:
        if "baseline" in steps:
            eval_baseline(
                model_name,
                tasks,
                n_shots,
                args.n_baseline,
                forward_batch_size_0_shot,
                forward_batch_size_n_shot,
                torch_seed,
                random_seed)
            flush()
        if "attributions" in steps:
            compute_attributions(
                model_name,
                tasks,
                n_shots,
                n_train,
                forward_batch_size_0_shot,
                forward_batch_size_n_shot,
                backward_batch_size,
                torch_seed,
                random_seed,
                robust=True)
            flush()
        if "compute_iterative" in steps:
            compute_iterative(
                model_name,
                tasks,
                sizes,
                n_train,
                backward_batch_size,
                args.max_iterations,
                args.tol,
                torch_seed,
                random_seed,
                n_restarts=args.n_restarts,
                noise_scale=args.noise_scale,
                )
            flush()
        if "eval_iterative" in steps:
            eval_iterative(
                model_name,
                tasks,
                n_test,
                forward_batch_size_0_shot,
                sizes,
                args.alpha,
                torch_seed,
                random_seed)
            flush()
        if "no_robustification" in steps:
            compute_attributions(
                model_name,
                tasks,
                n_shots,
                n_train,
                forward_batch_size_0_shot,
                forward_batch_size_n_shot,
                backward_batch_size,
                torch_seed,
                random_seed,
                robust=False)
            flush()
            eval_first_order(
                model_name,
                tasks,
                n_train,
                n_test,
                forward_batch_size_0_shot,
                sizes,
                scales,
                args.alpha,
                torch_seed,
                random_seed,
                robust=False)
            flush()
        if "no_refinement" in steps:
            eval_first_order(
                model_name,
                tasks,
                n_train,
                n_test,
                forward_batch_size_0_shot,
                sizes,
                scales,
                args.alpha,
                torch_seed,
                random_seed,
                robust=True)
            flush()
        if "prepare_figures" in steps:
            prepare_figures(model_name, tasks, sizes)
    if "intervention_effects" in steps:
        run_intervention_effects(torch_seed, random_seed)