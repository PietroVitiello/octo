"""
This script demonstrates how to finetune Octo to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).

To run this example, first download and extract the dataset from here: https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir=...
"""
from absl import logging
import flax
import jax
from jax.lib import xla_bridge
import optax
import tensorflow as tf
import tqdm
import wandb

from transformers import TFAutoModel

from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

def main():

    batch_size = 90
    freeze_transformer = False
    training_steps = 80000
    record_every = 200
    save_every = 2000

    use_wandb = True
    run_name = "octo_3" 
    save_dir = f"/home/pita/Documents/Projects/octo/models/{run_name}"
    is_original = True

    print(f"\n\nHardware acceleration: {xla_bridge.get_backend().platform}")

    print(f"Batch size: {batch_size}\n\n")

    assert (
        batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    # setup wandb for logging
    if use_wandb:
        wandb.init(
            project="RplusX",
            name=run_name,
            tags=["octo",]
        )

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    # pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)
    if is_original:
        pretrained_model = OctoModel.load_pretrained("octo-small-1.5")
    else:
        pretrained_model = OctoModel.load_pretrained("models/octo_1")
    # pretrained_model = TFAutoModel.from_pretrained("rail-berkeley/octo-small-1.5")
    # pretrained_model = TFAutoModel.from_pretrained("rail-berkeley/octo-small")

    print(f"\n\n\n{'-'*20} Model loaded {'-'*20}\n\n\n")

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    # TODO: directly load this from raw data to make it less opaque?
    logging.info("Loading finetuning dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="r_plus_x_dataset",
            data_dir=None,
            # data_dir="/home/pita/tensorflow_datasets/r_plus_x_dataset",
            image_obs_keys={"primary": "image"},
            # depth_obs_keys={"primary": "depth"},
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            action_horizon=1,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
            depth_resize_size={"primary": (256, 256)},
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)  # can reduce this if RAM consumption too high
        .batch(batch_size)
        .iterator()
    )

    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    # for key, value in example_batch.items():
    #     print(f"{key}: {value}")
    # print("\n")
    # for key, value in example_batch.items():
    #     print(f"{key}")
    # print("")
    # for key, value in example_batch["observation"].items():
    #     print(f"{key}: {value}")

    print(f"\n\n\n{'-'*20} Dataset loaded {'-'*20}\n\n\n")
    # print(f"dataset statistics:\n{dataset.dataset_statistics}")

    config = pretrained_model.config
    if is_original:
        # load pre-training config and modify --> remove wrist cam, add proprio input, change action head
        # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA
        del config["model"]["observation_tokenizers"]["wrist"]

        # Fully override the old action head with a new one (for smaller changes, you can use update_config)
        config["model"]["heads"]["action"] = ModuleSpec.create(
            L1ActionHead,
            action_horizon=1,
            action_dim=480,
            readout_key="readout_action",
        )

    # initialize weights for modified Octo model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model

    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    reference_lr = 5e-5
    minimum_lr = 1e-8
    warmup_steps = 200
    # learning_rate = optax.join_schedules(
    #     [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    # )
    learning_rate = optax.join_schedules(
        [
            optax.linear_schedule(0, reference_lr, warmup_steps),
            optax.linear_schedule(
                reference_lr, minimum_lr,
                training_steps - warmup_steps)
        ],
        [warmup_steps]
    )

    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # run finetuning loop
    logging.info("Starting finetuning...")
    for i in tqdm.tqdm(range(training_steps), total=training_steps, dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % record_every == 0:
            update_info = jax.device_get(update_info)
            if use_wandb:
                wandb.log(
                    flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                    step=i,
                )
        if (i + 1) % save_every == 0:
            # save checkpoint
            train_state.model.save_pretrained(step=i, checkpoint_path=save_dir)


if __name__ == "__main__":
    main()
