# Dynamics of collective mind in online news communities

![Project Image](Fig1.png)

## Description
This is a repository for the code used in "Dynamics of collective mind in online news communities" (Ha, 2025).
> Collective discourse and behaviors are shaped by the semantic representations of knowledge and beliefs shared by community members. This collective mind is susceptible to a variety of influences, from editorial practices (alignment, amplification, and reframing of news) to community dynamics (turnover, trolls, and counterspeech). It is critical that communities understand the effects of these influences so that they can protect themselves against manipulation and promote constructive discourse and behaviors. However, this understanding has been limited by the inability to conduct counterfactual experiments in the real world and by the inherent difficulty of predicting complex social systems. Here, we develop a computational model of collective minds, calibrated with data from 400 million comments posted on five online news platforms. The model provides a quantitative understanding of the way collective minds evolve in the context of continuously incoming news about outside events. It enables experimentation with different editorial and community influences, providing insights into the magnitude and persistence of their effects. Our results inform communities about the ways their collective mind can be influenced and what they can do to promote and sustain favorable collective dynamics.

## Installation
```sh
# Clone the repository
git clone https://github.com/nokpil/collmind.git

# Navigate into the project directory
cd collmind

# Install dependencies (if applicable)
pip install -r requirements.txt
```

## Usage
### Computational model
For given parameters, the simplest form of
```sh
python model.py --model_type {model_type} --topic_num {topic_num} --comm_num {comm_num} --event_num 1000 --event_topic_num 3 --filter_strength {filter_strength} --memory_strength {memory_strength} --timestep {timestep} --iv_rank {iv_rank} --init_type {init_type} --init_freq_std {init_freq_std} --folder {folder} --store_events {store_events} --store_extra {store_extra} --store_weight {store_weight} --store_corr {store_corr} --store_tmp {store_tmp} --desc {desc}
```
| Parameter            | Description                                      |
|----------------------|--------------------------------------------------|
| `--model_type`       | Type of the model (e.g., `A`, `B`, `C`)         |
| `--topic_num`        | Number of topics in the model                   |
| `--comm_num`         | Number of communities in the model              |
| `--event_num`        | Number of total events (default: `1000`)        |
| `--event_topic_num`  | Number of topics per event (default: `3`)       |
| `--filter_strength`  | Strength of the filtering mechanism             |
| `--memory_strength`  | Memory retention strength                       |
| `--timestep`         | Number of timesteps in simulation               |
| `--iv_rank`          | Rank of initial values                          |
| `--init_type`        | Initialization method                           |
| `--init_freq_std`    | Standard deviation of initialization frequency  |
| `--folder`           | Folder to store results                         |
| `--store_events`     | Whether to store event logs (`True` / `False`)  |
| `--store_extra`      | Whether to store extra data (`True` / `False`)  |
| `--store_weight`     | Whether to store weight data (`True` / `False`) |
| `--store_corr`       | Whether to store correlation data (`True` / `False`) |
| `--store_tmp`        | Whether to store temporary files (`True` / `False`) |
| `--desc`             | Short description of the experiment             |

For detailed usage for each influence simulation, check [intervention.ipynb](comp_model/intervention.ipynb).

### Data preprocessing
TBD

## Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are welcome.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Contact
For any inquiries, please contact seungwoong.ha@santafe.edu or create an issue.
