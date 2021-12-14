import copy
import os
import pickle
import sys
import time
from collections import OrderedDict, defaultdict

import torch

import numpy as np
from hyperopt import hp
import ray
import ray.tune
import gorilla
from ray.tune.trial import Trial
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import register_trainable
from tqdm import tqdm

from FastAutoAugment.archive import remove_deplicates, policy_decoder
from FastAutoAugment.augmentations import augment_list
from FastAutoAugment.common import get_logger, add_filehandler
from FastAutoAugment.data import get_dataloaders
from FastAutoAugment.metrics import Accumulator
from FastAutoAugment.networks import get_model, num_class
from FastAutoAugment.train import train_and_eval
from theconf import Config as C, ConfigArgumentParser


top1_valid_by_cv = defaultdict(lambda: list)


def step_w_log(self):
    original = gorilla.get_original_attribute(ray.tune.trial_runner.TrialRunner, "step")

    # log
    cnts = OrderedDict()
    for status in [
        Trial.RUNNING,
        Trial.TERMINATED,
        Trial.PENDING,
        Trial.PAUSED,
        Trial.ERROR,
    ]:
        cnt = len(list(filter(lambda x: x.status == status, self._trials)))
        cnts[status] = cnt
    best_top1_acc = 0.0
    for trial in filter(lambda x: x.status == Trial.TERMINATED, self._trials):
        if not trial.last_result:
            continue
        best_top1_acc = max(best_top1_acc, trial.last_result["top1_valid"])
    print("iter", self._iteration, "top1_acc=%.3f" % best_top1_acc, cnts, end="\r")
    return original(self)


# gorilla.Patch(destination, name, object, settings)
# allow_hit: allow setting a new value for an existing attribute.
patch = gorilla.Patch(
    ray.tune.trial_runner.TrialRunner,
    "step",
    step_w_log,
    settings=gorilla.Settings(allow_hit=True),
)
gorilla.apply(patch)


logger = get_logger("Fast AutoAugment")


def _get_path(dataset, model, tag, basepath="/tmp"):
    return os.path.join(
        basepath,
        "%s_%s_%s.model" % (dataset, model, tag),
    )


@ray.remote(num_cpus=4, num_gpus=1, max_calls=1)
def train_model(
    config, dataroot, augment, cv_ratio_test, cv_fold, save_path=None, skip_exist=False
):
    logger.info("cuda: %s", torch.cuda.is_available())
    logger.info(
        "train_model: "
        "config=%s, dataroot=%s, augment=%s, cv_ratio_test=%s, cv_fold=%s, "
        "save_path=%s, skip_exist=%s",
        config,
        dataroot,
        augment,
        cv_ratio_test,
        cv_fold,
        save_path,
        skip_exist,
    )
    C.get()
    C.get().conf = config
    C.get()["aug"] = augment

    result = train_and_eval(
        None,
        dataroot,
        cv_ratio_test,
        cv_fold,
        save_path=save_path,
        only_eval=skip_exist,
    )
    return C.get()["model"]["type"], cv_fold, result


def eval_tta(config, augment, checkpoint_dir=None):
    """Evaluate test-time augmentation"""
    C.get()
    C.get().conf = config
    cv_ratio_test, cv_fold, save_path = (
        augment["cv_ratio_test"],
        augment["cv_fold"],
        augment["save_path"],
    )

    # setup - provided augmentation rules
    C.get()["aug"] = policy_decoder(augment, augment["num_policy"], augment["num_op"])

    # get_model(model_name, dataset_name)
    model = get_model(C.get()["model"], num_class(C.get()["dataset"]))
    ckpt = torch.load(save_path)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # each loader is of type torch.utils.data.dataloader._MultiProcessingDataLoaderIter
    loaders = []
    for _ in range(augment["num_policy"]):  # TODO
        _, tl, validloader, tl2 = get_dataloaders(
            C.get()["dataset"],
            C.get()["batch"],
            augment["dataroot"],
            cv_ratio_test,
            split_idx=cv_fold,
        )
        loaders.append(iter(validloader))
        del tl, tl2

    start_t = time.time()
    metrics = Accumulator()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    # TODO jr: does it iterate through all the loaders?
    #          or does it stop after exhausting just the first loader?
    try:
        while True:
            losses = []
            corrects = []
            for loader in loaders:
                data, label = next(loader)  # StopIteration is thrown here
                data = data.cuda()
                label = label.cuda()

                pred = model(data)

                loss = loss_fn(pred, label)
                losses.append(loss.detach().cpu().numpy())

                _, pred = pred.topk(1, 1, True, True)
                pred = pred.t()
                correct = (
                    pred.eq(label.view(1, -1).expand_as(pred)).detach().cpu().numpy()
                )
                corrects.append(correct)
                del loss, correct, pred, data, label

            losses = np.concatenate(losses)
            losses_min = np.min(losses, axis=0).squeeze()

            corrects = np.concatenate(corrects)
            corrects_max = np.max(corrects, axis=0).squeeze()
            metrics.add_dict(
                {
                    "minus_loss": -1 * np.sum(losses_min),
                    "correct": np.sum(corrects_max),
                    "cnt": len(corrects_max),
                }
            )
            del corrects, corrects_max
    except StopIteration:
        pass

    del model
    metrics = metrics / "cnt"
    gpu_secs = (time.time() - start_t) * torch.cuda.device_count()
    ray.tune.report(
        minus_loss=metrics["minus_loss"],
        top1_valid=metrics["correct"],
        elapsed_time=gpu_secs,
        done=True,
    )
    return metrics["correct"]


if __name__ == "__main__":
    import json
    from pystopwatch2 import PyStopwatch

    w = PyStopwatch()

    parser = ConfigArgumentParser(conflict_handler="resolve")
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/data/private/pretrainedmodels",
        help="torchvision data folder",
    )
    parser.add_argument("--until", type=int, default=5)
    parser.add_argument(
        "--num-op", type=int, default=2, help="Number of operations per policy"
    )
    parser.add_argument("--num-policy", type=int, default=5)
    parser.add_argument(
        "--num-search",
        type=int,
        default=200,
        help="Number of times to sample from the hyperparameter space."
        "  Set to 4 if smoke-test",
    )
    parser.add_argument("--cv-ratio", type=float, default=0.4)
    parser.add_argument(
        "--decay", type=float, default=-1, help="Value of lambda for L2 regularization"
    )
    # parser.add_argument("--redis", type=str, default="gpu-cloud-vnode30.dakao.io:23655")
    parser.add_argument("--per-class", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", "-o", type=str, default="/tmp")
    args = parser.parse_args()

    logger.info("args: %s", args)

    if args.decay > 0:
        logger.info("decay=%.4f" % args.decay)
        C.get()["optimizer"]["decay"] = args.decay

    if args.smoke_test:
        if args.num_search > 4:
            args.num_search = 4
            logger.info("num_search set to 4 for smoke-test")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    add_filehandler(
        logger,
        os.path.join(
            args.output_dir,
            "%s_%s_cv%.1f.log"
            % (C.get()["dataset"], C.get()["model"]["type"], args.cv_ratio),
        ),
    )
    logger.info("configuration...")
    logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4))

    # -----------------------------------------------------------------------
    # 0. Initialize Ray
    # -----------------------------------------------------------------------
    logger.info("initialize ray...")
    ray.init()

    # ray.init(local_mode=True)
    # logger.info("runtime_env: %s", env)
    # ray.init("ray://localhost:10001", runtime_env=env)
    # ray.init("ray://localhost:10001")

    # -----------------------------------------------------------------------
    # 1. Train models (without augmentation)
    # -----------------------------------------------------------------------
    num_result_per_cv = 10
    cv_num = 5
    copied_c = copy.deepcopy(C.get().conf)

    logger.info(
        "search augmentation policies, dataset=%s model=%s"
        % (C.get()["dataset"], C.get()["model"]["type"])
    )

    logger.info(
        "----- Train without Augmentations cv=%d ratio(test)=%.1f -----"
        % (cv_num, args.cv_ratio)
    )

    w.start(tag="train_no_aug")
    paths = [
        _get_path(
            C.get()["dataset"],
            C.get()["model"]["type"],
            "ratio%.1f_fold%d" % (args.cv_ratio, i),
            basepath=args.output_dir,
        )
        for i in range(cv_num)
    ]
    logger.info("paths: %s", paths)
    reqs = [
        train_model.remote(
            copy.deepcopy(copied_c),
            args.dataroot,
            C.get()["aug"],
            args.cv_ratio,
            i,
            save_path=paths[i],
            skip_exist=True,
        )
        for i in range(cv_num)
    ]

    tqdm_epoch = tqdm(range(C.get()["epoch"]))
    logger.info("num epochs: %s", tqdm_epoch)
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs_per_cv = OrderedDict()
            for cv_idx in range(cv_num):
                try:
                    latest_ckpt = torch.load(paths[cv_idx])
                    if "epoch" not in latest_ckpt:
                        epochs_per_cv["cv%d" % (cv_idx + 1)] = C.get()["epoch"]
                        continue
                    epochs_per_cv["cv%d" % (cv_idx + 1)] = latest_ckpt["epoch"]
                except Exception as e:
                    logger.warning("%s: %s", e.__class__.__name__, e)
                    time.sleep(2)
                    continue
            tqdm_epoch.set_postfix(epochs_per_cv)
            if (
                len(epochs_per_cv) == cv_num
                and min(epochs_per_cv.values()) >= C.get()["epoch"]
            ):
                is_done = True
            if len(epochs_per_cv) == cv_num and min(epochs_per_cv.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break

    # --------------------
    logger.info("getting results...")
    pretrain_results = ray.get(reqs)
    for r_model, r_cv, r_dict in pretrain_results:
        logger.info(
            "model=%s cv=%d top1_train=%.4f top1_valid=%.4f"
            % (r_model, r_cv + 1, r_dict["top1_train"], r_dict["top1_valid"])
        )
    logger.info("processed in %.4f secs" % w.pause("train_no_aug"))

    if args.until == 1:
        sys.exit(0)

    # -----------------------------------------------------------------------
    # 2. Search augmentation policies
    # -----------------------------------------------------------------------
    logger.info("----- Search Test-Time Augmentation Policies -----")
    w.start(tag="search")

    ops = augment_list(False)
    space = {}
    for i in range(args.num_policy):
        for j in range(args.num_op):
            space["policy_%d_%d" % (i, j)] = hp.choice(
                "policy_%d_%d" % (i, j), list(range(0, len(ops)))
            )
            space["prob_%d_%d" % (i, j)] = hp.uniform("prob_%d_ %d" % (i, j), 0.0, 1.0)
            space["level_%d_%d" % (i, j)] = hp.uniform(
                "level_%d_ %d" % (i, j), 0.0, 1.0
            )

    final_policy_set = []
    total_computation = 0
    reward_attr = "top1_valid"  # top1_valid or minus_loss
    logger.info("reward_attr: %s", reward_attr)

    for _ in range(1):  # run multiple times.
        for cv_fold in range(cv_num):
            name = "search_%s_%s_fold%d_ratio%.1f" % (
                C.get()["dataset"],
                C.get()["model"]["type"],
                cv_fold,
                args.cv_ratio,
            )
            logger.info("trainable: %s", name)
            # checkpoint_dir=None prevents checkpointing from being removed.  Warning without it:
            # WARNING function_runner.py:561 -- Function checkpointing is disabled. This may result
            # in unexpected behavior when using checkpointing features or certain schedulers.
            # To enable, set the train function arguments to be `func(config, checkpoint_dir=None)`.
            register_trainable(
                name,
                lambda augs, checkpoint_dir=None: eval_tta(
                    copy.deepcopy(copied_c), augs, checkpoint_dir=checkpoint_dir
                ),
            )

            # jr notes for ray v1.9.0
            # max_concurrent is deprecated.
            #   DeprecationWarning: `max_concurrent` is deprecated for this search
            #   algorithm.  Use tune.suggest.ConcurrencyLimiter() instead. This will
            #   raise an error in future versions of Ray.
            # `reward_attr`` was removed
            # algo = HyperOptSearch(space, max_concurrent=4 * 20, reward_attr=reward_attr)
            algo = HyperOptSearch(space, metric=reward_attr, mode="min")

            config = {
                "dataroot": args.dataroot,
                "save_path": paths[cv_fold],
                "cv_ratio_test": args.cv_ratio,
                "cv_fold": cv_fold,
                "num_op": args.num_op,
                "num_policy": args.num_policy,
            }
            # jr: removed scheduler=None, queue_trials=True (deprecated)
            analysis = ray.tune.run(
                name,
                config=config,
                search_alg=algo,
                num_samples=args.num_search,
                resources_per_trial={"gpu": 1, "cpu": 4},
                stop={"training_iteration": args.num_policy},
                verbose=1,
                resume=args.resume,
                raise_on_failed_trial=False,
                # max_concurrent=4 * 20,
            )

            logger.info("trial stats: %s", analysis.stats())

            with open(os.path.join(args.output_dir, "results.pickle"), "wb") as fobj:
                pickle.dump(
                    {
                        "trial_dataframes": analysis.trial_dataframes,
                        "results": analysis.results,
                        "results_df": analysis.results_df,
                    },
                    fobj,
                )

            results = analysis.trials
            # results is of type ray.tune.ExperimentAnalysis
            # https://docs.ray.io/en/latest/tune/api_docs/analysis.html#ray.tune.ExperimentAnalysis
            results = [
                x for x in results if x.last_result
            ]  # x.last_result is a dict or None
            results = sorted(
                results, key=lambda x: x.last_result[reward_attr], reverse=True
            )

            # calculate computation usage
            for result in results:
                total_computation += result.last_result["elapsed_time"]

            for result in results[:num_result_per_cv]:
                final_policy = policy_decoder(
                    result.config, args.num_policy, args.num_op
                )
                logger.info(
                    "minus_loss=%.12f top1_valid=%.4f policy=%s"
                    % (
                        result.last_result["minus_loss"],
                        result.last_result["top1_valid"],
                        final_policy,
                    )
                )

                final_policy = remove_deplicates(final_policy)
                final_policy_set.extend(final_policy)

    logger.info("final_policy length=%d" % len(final_policy_set))
    logger.info(json.dumps(final_policy_set))
    logger.info(
        "processed in %.4f secs, gpu hours=%.4f"
        % (w.pause("search"), total_computation / 3600.0)
    )

    # -----------------------------------------------------------------------
    # 3. Train full with baseline augmentation and found policies.
    # -----------------------------------------------------------------------
    logger.info(
        "----- Train with Augmentations model=%s dataset=%s aug=%s ratio(test)=%.1f -----"
        % (C.get()["model"]["type"], C.get()["dataset"], C.get()["aug"], args.cv_ratio)
    )
    w.start(tag="train_aug")

    num_experiments = 5
    logger.info("num_experiments=%s", num_experiments)

    default_path = [
        _get_path(
            C.get()["dataset"],
            C.get()["model"]["type"],
            "ratio%.1f_default%d" % (args.cv_ratio, _),
            basepath=args.output_dir,
        )
        for _ in range(num_experiments)
    ]
    augment_path = [
        _get_path(
            C.get()["dataset"],
            C.get()["model"]["type"],
            "ratio%.1f_augment%d" % (args.cv_ratio, _),
            basepath=args.output_dir,
        )
        for _ in range(num_experiments)
    ]
    reqs = [
        train_model.remote(
            config=copy.deepcopy(copied_c),
            dataroot=args.dataroot,
            augment=C.get()["aug"],
            cv_ratio_test=0.0,
            cv_fold=0,
            save_path=default_path[i],
            skip_exist=True,
        )
        for i in range(num_experiments)
    ] + [
        train_model.remote(
            config=copy.deepcopy(copied_c),
            dataroot=args.dataroot,
            augment=final_policy_set,
            cv_ratio_test=0.0,
            cv_fold=0,
            save_path=augment_path[i],
        )
        for i in range(num_experiments)
    ]

    tqdm_epoch = tqdm(range(C.get()["epoch"]))
    is_done = False
    for epoch in tqdm_epoch:
        while True:
            epochs = OrderedDict()
            for exp_idx in range(num_experiments):
                try:
                    if os.path.exists(default_path[exp_idx]):
                        latest_ckpt = torch.load(default_path[exp_idx])
                        epochs["default_exp%d" % (exp_idx + 1)] = latest_ckpt["epoch"]
                except Exception:
                    pass
                try:
                    if os.path.exists(augment_path[exp_idx]):
                        latest_ckpt = torch.load(augment_path[exp_idx])
                        epochs["augment_exp%d" % (exp_idx + 1)] = latest_ckpt["epoch"]
                except Exception:
                    pass

            tqdm_epoch.set_postfix(epochs)
            if (
                len(epochs) == num_experiments * 2
                and min(epochs.values()) >= C.get()["epoch"]
            ):
                is_done = True
            if len(epochs) == num_experiments * 2 and min(epochs.values()) >= epoch:
                break
            time.sleep(10)
        if is_done:
            break

    logger.info("getting results...")
    final_results = ray.get(reqs)

    for train_mode in ["default", "augment"]:
        avg = 0.0
        for _ in range(num_experiments):
            r_model, r_cv, r_dict = final_results.pop(0)
            logger.info(
                "[%s] top1_train=%.4f top1_test=%.4f"
                % (train_mode, r_dict["top1_train"], r_dict["top1_test"])
            )
            avg += r_dict["top1_test"]
        avg /= num_experiments
        logger.info(
            "[%s] top1_test average=%.4f (#experiments=%d)"
            % (train_mode, avg, num_experiments)
        )
    logger.info("processed in %.4f secs" % w.pause("train_aug"))

    logger.info(w)
