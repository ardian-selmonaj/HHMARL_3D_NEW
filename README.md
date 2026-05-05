# Hierarchical Multi-Agent Reinforcement Learning for simulated Air Combat
3D Heterogeneous Hierarchical MARL for realistic air combat simulation. This is the official implementation of [this paper](https://arxiv.org/pdf/2510.11474).

## Training Procedure
The hierarchical structure in our method is organized by first training low-level control policies, then fixing these policies and providing them as options to train the high-level commander. For this, run `train.py` with `--hierarchy_level=0` for low-level training and `--hierarchy_level=1` to train the high-level policy (commander). The low-level policies *must* be pre-trained and stored in order to start training of the commander policy. Storing and correct naming of the folder is automatically done. The naming of the policies in the code differes from the namings in the [paper](https://arxiv.org/pdf/2510.11474):
- attack = cannon
- engage = track
- defend = escape
- commander = hier

### Training Parameters
Most important configuration arguments are:
- `mission` is either "cannon", "track" or "escape" for the three low-level policies
- `level` is the level to train low-level policies. Ranges from `[1-4]` for "cannon", "track" and `[1-2]` for "escape". 
- for `level >= 2` you have to set `restore=True` to restore the model from the previous level. 
- `hierarchy_level` is either `0` (low-level) or `1` (high-level, commander).
- `model_name` to specify aircraft type, either "f16" (default) or "A4". See second point of the step-by-step instructions for more details. 

All configuration arguments can be found in `base_config.py` within 'configs' folder. Within this folder, there are specific `.yaml` files for the respective mission, e.g., specificying aircraft type (model), map size, rewards, etc. 

### Step-by-Step Training
1) run `train.py` with `--mission` as "cannon" and "track" for levels [1-4] each. When training with `--level=1` completed, continue with `--level=2` and `--restore=True` with the same mission. Do the same for `--mission="escape"` for levels 1 and 2. Do not forget to restore!

2) if *heterogeneous* setup is desired, repeat first step with `--model_name="A4"` in `base_config.py`. *You then also need to change* the `model` name in the respective `.yaml` file `L*_"mission".yaml` to `model: "A4"`, otherwise JSBSim still uses the "f16" as specified in the `.yaml` file. In the current version of JSBsim, we recommend using only f16 and A4 models. If *homogeneous* setup is enough for you, then you can skip this step, so only f16 aircraft models are used. `model_name` in `base_config.py` is used for correct naming when storing results, while the `model` argument in `.yaml` files is used for JSBSim. This structure will be improved in the future. 

3) check that results are stored correctly in 'results' folder with the naming structure as `L*_"mission_"model"`, e.g., "L1_cannon_f16". This is the same naming as the ´.yaml´ mission files. Make sure to proceed as described in step 2 for the *heterogeneous* training and storing. 

4) when all low-level policies are trained and stored, then continue with `train.py` and `--hierarchy_level=1`. Restoring, levels and specifying missions are not required anymore. The `build_rl_modules()` within `train.py` automatically learns two high-level policies, "f16" and "A4". But in the *homogeneous* case (using only "f16"), the "A4" policy won't learn as it doesn't receive observations. You can go to `missions > tasks > task_hier.py` to see which low-level policies are expected in `pol_dirs` under the method `load_model()`. Check that these namings exist and correspond to the ones described under step 3. 

5) when training of all policies completes, you can do inference using `test.py` for low-level policies and `test_hier.py` for high-level. The latter also evaluates combats by number of wins, draws and losses. You can render the situation through our PyGame visualizer.

**Note:** The shooting penalty of -0.1 was slightly adjusted so that it is now applied only when the cannon is actually fired, rather than when the shoot action is selected. As a result, the initial results differ slightly from those reported in the paper, but the model ultimately converges in a similar way.

### GPU training
When training on GPU, specified by argument `num_learners_gpus=1`, then a slight adjustment in the source-code of Ray is required in order to restore the model later. Go to `ray>rllib>core>learner>torch`, open `torch_learner.py`, and extend line 224 as:
```
...
optim = self.get_optimizer(module_id, optimizer_name)  #given
for g in optim.param_groups:  #extended
  g["capturable"] = True  #extended
...
```

## Hierarchical Observation
The parameter `N_OBS` within `networks.py` determines the observation of the hierarchical actor (policy). By default `N_OBS=2`, i.e., the hierarchical commander observes two closest opponent aircraft. Additionally, the code is configured for *symmetric n-vs-n setups*, i.e., 3-vs-3, 5-vs-5, etc. The team sizes can be set in `hier.yaml` mission configuration. This is determined by `num_rl_agents` and `num_opp_agents`, which must match the number of specified aircraft. When changing the team sizes, you must modify `N_TEAM` in `networks.py` accordingly, i.e. `N_TEAM = num_opp_agents` must be the same. When trainig the hierarchical commander, you are free to set `N_OBS` to change the radar range of the commander, but keep in mind that `N_OBS <= num_opp_agents` must hold, otherwise an error is thrown, as the commander cannot observe more opponents as are present!

## Simple Policy Optimization
SPO is not implemented in Ray, but can be done manually in source code of Ray RLlib. Go to: `ray > rllib > algorithms > ppo > torch > ppo_torch_learner.py` and modify `surrogate loss` to 

`surrogate_loss = batch[Postprocessing.ADVANTAGES] * logp_ratio - (torch.abs(batch[Postprocessing.ADVANTAGES])/(2*config.clip_param)) * (logp_ratio-1)**2`.

## Required Packages
These packages are provided in ´requirements.txt´
- gymnasium>=0.26.3
- jsbsim==1.2.1
- moderngl==5.10.0
- pandas>=2.2.0
- pillow==11.0.0
- protobuf==5.29.1
- psutil==6.1.0
- pygame>=2.6.1
- pymap3d>=3.1.0
- tensorboard==2.18.0
- torch==2.7.0
- tqdm>=4.67.0
- wandb==0.18.7
- numpy==2.2.6
- ray["rllib"]==2.48.0

**IMPORTANT** DO NOT INSTALL pydantic, as it raises an error in rendering!


## Acknowledgements and Citation
We appreciate the work done by the authors of the [LightAircraftGame (LAG)](https://github.com/liuqh16/LAG), which is open-source as our project. Our code makes use of the following parts of the LAG environment: 1) JSBSim catalog with similar structure in ´sim.py´, 2) similar structure of mission configuration with ´.yaml´ files, and 3) Pre-trained model ´baseline_actor.pt´ within ´missions>models´. Even thouth the baseline actor could be trained in this step, it is employed directly from LAG to speed up training. Apart from this, our environment and hierarchical structure is different to LAG. 

To cite our work, please proceed as:
```
@inproceedings{hhmarl3d,
  title={Coordinated Strategies in Realistic Air Combat by Hierarchical Multi-Agent Reinforcement Learning},
  author={Selmonaj, Ardian and Del Rio, Giacomo and Schneider, Adrian and Antonucci, Alessandro},
  booktitle={2025 IEEE International Conference on Agentic AI (ICA)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```



# DEVELOPMENT
DEVELOPMENT BRANCH, testing new features
### JSBsim values
- `v_down` is positive when flying towards earth, and negative when increasing altitude. 
- Psi = Yaw (heading)
- Theta = Pitch (eleveation)
- Phi = Roll (tilt)