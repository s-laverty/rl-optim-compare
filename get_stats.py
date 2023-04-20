import utils
import json

# Parse config file.
with open('config/perf-results.json') as f:
    config = utils.Config(**json.load(f))

iteration = 1600
ranks = (0, 2, 4, 5)

# Get results
path = utils.checkpoint_path(config, 5)
episode_lengths = {
    name:[
        utils.load_checkpoint(path.joinpath(utils.checkpoint_name(i, rank))) \
            [5][iteration]
        for i in range(config['num_models'])
    ]
    for name, rank in zip(
        (config['optimizers'][rank]['name'] for rank in ranks),
        ranks,
    )
}
print(episode_lengths)
