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
Move to the 'comp_model' directory and run the following code.
```sh
python model.py --model_type {model_type} --topic_num {topic_num} --comm_num {comm_num} --event_num 1000 --event_topic_num 3 --filter_strength {filter_strength} --memory_strength {memory_strength} --timestep {timestep} --iv_rank {iv_rank} --init_type {init_type} --init_freq_std {init_freq_std} --folder {folder} --store_events {store_events} --store_extra {store_extra} --store_weight {store_weight} --store_corr {store_corr} --store_tmp {store_tmp} --desc {desc}
```
| Parameter            | Description                                      |
|----------------------|--------------------------------------------------|
| `--model_type`       | Type of the model (keep this to 'simple')         |
| `--topic_num`        | Number of topics in the model (default: `200`)                   |
| `--comm_num`         | Number of communities in the model (default: `5`)             |
| `--event_num`        | Number of total events (default: `1000`)        |
| `--event_topic_num`  | Number of topics per event (default: `3`)       |
| `--filter_strength`  | Strength of the filtering mechanism (`$0 \geq \lambda_m$`)             |
| `--memory_strength`  | Memory retention strength  (`$0 \geq \lambda_m<1$`)                     |
| `--timestep`         | Number of timesteps in simulation               |
| `--iv_type`          | Influence type (1: Alignment, 2:Amplification, 3: Reframing, 4: Membership turnover, 5: Trolls, 6: Counterspeech)|
| `--iv_t1`          | Influence time (1)                         |
| `--iv_s1`          | Influence strength (1)                          |
| `--iv_t2`          | Influence time (2)                          |
| `--iv_s2`          | Influence strength (2)                          |
| `--iv_rank`          | Influence rank                        |
| `--iv_tier`          | Influence tier                         |
| `--init_type`        | Initialization method (`plain`, `perturb`, `fixed`)                           |
| `--init_freq_std`    | Standard deviation of initialization frequency (default: `0.0`) |
| `--init_weight_std`        |Standard deviation of initialization weight (default: `0.0`)  |
| `--folder_name`           | Subfolder name to store results                     |
| `--store_events`     | Whether to store event logs (`True` / `False`)  |
| `--store_extra`      | Whether to store extra data (`True` / `False`)  |
| `--store_weight`     | Whether to store weight data (`True` / `False`) |
| `--store_corr`       | Whether to store correlation data (`True` / `False`) |
| `--desc`             | Short description of the experiment             |

* Influence time and strength typically means the start and end of a certain influence, but one may use this to freely control the influence strength at iv_t1 and iv_t2.
* Simulation results will be stored under `model_results/folder_name`.
* To use `fixed` settings for init_type, one needs to first construct `fixed_dict.pkl` under `comp_model/data`, which can be done by the code snippet in the [nb_prepreocess.ipynb](nb_preprocess.ipynb).

For more detailed usage for each influence simulation, check [intervention.ipynb](comp_model/intervention.ipynb).

### Data preprocessing
TBD

## Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are welcome.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Contact
For any inquiries, please contact seungwoong.ha@santafe.edu or create an issue.
