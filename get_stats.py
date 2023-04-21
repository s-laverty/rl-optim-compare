import utils
import json
import numpy as np

# Parse config file.
with open('config/perf-results.json') as f:
    config = utils.Config(**json.load(f))

# iteration = 1600
ranks = (0, 2, 4, 5)

# Get results
path = utils.checkpoint_path(config, 5)
episode_lengths = np.asarray([
    [
        utils.load_checkpoint(path.joinpath(
            utils.checkpoint_name(rank, optim_rank)))[5]
        for rank in range(config['num_models'])
    ]
    for optim_rank in ranks
])
mean_loss = episode_lengths.mean(axis=1)
best_results = {
    name: list(zip(best_iterations, best_evals))
    for name, best_iterations, best_evals in zip(
        (config['optimizers'][optim_rank]['name'] for optim_rank in ranks),
        np.argmax(episode_lengths, axis=-1),
        np.max(episode_lengths, axis=-1),
    )
}
best_avg_results = {
    name: (best_iteration, evals[:,best_iteration])
    for name, best_iteration, evals in zip(
        (config['optimizers'][optim_rank]['name'] for optim_rank in ranks),
        np.argmax(mean_loss, axis=-1),
        episode_lengths,
    )
}
# print(best_results)
print(best_avg_results)
