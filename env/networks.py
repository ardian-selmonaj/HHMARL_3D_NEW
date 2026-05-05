
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.core.columns import Columns
import numpy as np

from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
import torch.nn as nn

"""
Neural Network classes used during training.

These classes are implemented in a similar way as the following:

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
which is the same as
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule

and also the following:
from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
"""

AGENT_OBS = 17
OPP_OBS = 13

class CombatModel(TorchRLModule, ValueFunctionAPI):
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = PPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(RLModule)
    def setup(self):

        #self.model_config #is specified in AlgorithmConfig
        #self.pi = self.catalog.build_pi_head(framework=self.framework)

        self.inp1 = SlimFC(AGENT_OBS, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(OPP_OBS, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp3 = SlimFC(400, 300, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)

        required_output_dim = self.action_dist_cls.required_input_dim(space=self.action_space, model_config={})
        self.act_out = SlimFC(300,required_output_dim, activation_fn= None, initializer=torch.nn.init.orthogonal_)

        self.inp1_val = SlimFC(AGENT_OBS, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2_val = SlimFC(OPP_OBS, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp3_val = SlimFC(400, 300, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.val_out = SlimFC(300, 1, activation_fn= None, initializer=torch.nn.init.orthogonal_)

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        i1 = batch[Columns.OBS][:, :AGENT_OBS]
        i2 = batch[Columns.OBS][:, AGENT_OBS:]
        x = torch.cat((self.inp1(i1), self.inp2(i2)), dim=1)
        x = self.inp3(x)
        x = self.act_out(x)
        return {Columns.ACTION_DIST_INPUTS: x}

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        i1 = batch[Columns.OBS][:, :AGENT_OBS]
        i2 = batch[Columns.OBS][:, AGENT_OBS:]
        x = torch.cat((self.inp1_val(i1), self.inp2_val(i2)), dim=1)
        x = self.inp3_val(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])  # or .squeeze(-1)


class CombatModelAtt(TorchRLModule, ValueFunctionAPI):
    """
    Removed 'concat' at last step, so output layer takes in 200. Previously, layers were 130 -> 390 last layer input.
    -> in fact: switched from concat to skip connection.
    -> two Attention Heads.
    """
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = PPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(RLModule)
    def setup(self):

        #self.model_config #is specified in AlgorithmConfig
        #self.pi = self.catalog.build_pi_head(framework=self.framework)
        #tokens = obs.view(obs.shape[0], 4, 4) transform from 16D to 4 times 4D 

        self.inp1 = SlimFC(AGENT_OBS, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(OPP_OBS, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)

        self.act_att = nn.MultiheadAttention(200,2,batch_first=True)
        self.act_norm = nn.LayerNorm(200)

        required_output_dim = self.action_dist_cls.required_input_dim(space=self.action_space, model_config={})
        self.act_out = SlimFC(200,required_output_dim, activation_fn= None, initializer=torch.nn.init.orthogonal_)

        self.inp1_val = SlimFC(AGENT_OBS, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2_val = SlimFC(OPP_OBS, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)

        self.val_att = nn.MultiheadAttention(200,2,batch_first=True)
        self.val_norm = nn.LayerNorm(200)

        self.val_out = SlimFC(200, 1, activation_fn= None, initializer=torch.nn.init.orthogonal_)


    @override(RLModule)
    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS]
        i1 = obs[:, :AGENT_OBS]
        i2 = obs[:, AGENT_OBS:]

        x1 = self.inp1(i1)
        x2 = self.inp2(i2)

        x_emb = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        att, _ = self.act_att(x_emb, x_emb, x_emb, need_weights=False)
        att = self.act_norm(x_emb + att).mean(dim=1)
        x = self.act_out(att)

        #x = torch.cat([x1, x2, att], dim=1)
        # x = self.act_out(x)
        return {Columns.ACTION_DIST_INPUTS: x}

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        i1 = batch[Columns.OBS][:, :AGENT_OBS]
        i2 = batch[Columns.OBS][:, AGENT_OBS:]

        x1 = self.inp1_val(i1)
        x2 = self.inp2_val(i2)

        x_emb = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        att, _ = self.val_att(x_emb, x_emb, x_emb, need_weights=False)
        att = self.val_norm(x_emb + att).mean(dim=1)
        x = self.val_out(att)

        #att = self.val_norm(att).mean(dim=1)
        #x = torch.cat([x1, x2, att], dim=1)
        # x = self.val_out(x)
        return x.squeeze(-1)


class CombatModelAttSh(TorchRLModule, ValueFunctionAPI):
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = PPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(RLModule)
    def setup(self):

        self.inp1 = SlimFC(AGENT_OBS, 130, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(OPP_OBS, 130, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.act_att = nn.MultiheadAttention(130,1,batch_first=True)
        self.act_norm = nn.LayerNorm(390)

        required_output_dim = self.action_dist_cls.required_input_dim(space=self.action_space, model_config={})
        self.act_out = SlimFC(200,required_output_dim, activation_fn= None, initializer=torch.nn.init.orthogonal_)

        self.inp1_val = SlimFC(AGENT_OBS, 130, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2_val = SlimFC(OPP_OBS, 130, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.val_att = nn.MultiheadAttention(130,1,batch_first=True)
        self.val_norm = nn.LayerNorm(390)

        self.val_out = SlimFC(200, 1, activation_fn= None, initializer=torch.nn.init.orthogonal_)

        self.shared = SlimFC(390, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        i1 = batch[Columns.OBS][:, :AGENT_OBS]
        i2 = batch[Columns.OBS][:, AGENT_OBS:]

        x1 = self.inp1(i1)
        x2 = self.inp2(i2)

        x_emb = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        att, _ = self.act_att(x_emb, x_emb, x_emb, need_weights=False)
        att = att.mean(dim=1)

        x = torch.cat([x1, x2, att], dim=1)
        x = self.act_norm(x)
        x = self.shared(x)
        x = self.act_out(x)
        return {Columns.ACTION_DIST_INPUTS: x}

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        i1 = batch[Columns.OBS][:, :AGENT_OBS]
        i2 = batch[Columns.OBS][:, AGENT_OBS:]

        x1 = self.inp1_val(i1)
        x2 = self.inp2_val(i2)

        x_emb = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        att, _ = self.val_att(x_emb, x_emb, x_emb, need_weights=False)
        att = att.mean(dim=1)

        x = torch.cat([x1, x2, att], dim=1)
        x = self.val_norm(x)
        x = self.shared(x)
        x = self.val_out(x)
        return x.squeeze(-1)


class CombatModelGRU(TorchRLModule, ValueFunctionAPI):
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = PPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(TorchRLModule)
    def get_initial_state(self):
        return {
            "h": np.zeros(shape=(100,), dtype=np.float32),
        }

    @override(RLModule)
    def setup(self):

        self.inp1 = SlimFC(AGENT_OBS, 150, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(OPP_OBS, 150, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)

        self.act_gru = nn.GRU(AGENT_OBS+OPP_OBS, 100, batch_first=True)
        self.act_norm = nn.LayerNorm(100)

        required_output_dim = self.action_dist_cls.required_input_dim(space=self.action_space, model_config={})
        self.act_out = SlimFC(400,required_output_dim, activation_fn= None, initializer=torch.nn.init.orthogonal_)

        self.inp1_val = SlimFC(AGENT_OBS, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2_val = SlimFC(OPP_OBS, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.val_out = SlimFC(400, 1, activation_fn= None, initializer=torch.nn.init.orthogonal_)

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS]

        x1 = self.inp1(obs[:, :, :AGENT_OBS])
        x2 = self.inp2(obs[:, :, AGENT_OBS:])

        h = batch[Columns.STATE_IN]["h"]
        x3, h = self.act_gru(obs, h.unsqueeze(0))
        x3 = self.act_norm(x3)

        x = torch.cat([x1, x2, x3], dim=2)
        x = self.act_out(x)
        return {Columns.ACTION_DIST_INPUTS: x, Columns.STATE_OUT:{"h": h.squeeze(0)}}

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        i1 = batch[Columns.OBS][:, :, :AGENT_OBS]
        i2 = batch[Columns.OBS][:, :, AGENT_OBS:]
        x = torch.cat((self.inp1_val(i1), self.inp2_val(i2)), dim=2)
        x = self.val_out(x)
        return x.squeeze(-1)

HL_OWN = 8
HL_FRI = 6
HL_OPP = 6
N_OBS = 3 #how many opponents to observe
N_TEAM = 3 #team size

"""
Hierarchical observation is designed as:
actor (policy) observes own info + N_OBS opponents + 1 friendly aircraft
critic (value) observes full state: own + all opponents + all friendlies

--> NOTE: configured for symmetrics n-vs-n setups!
"""

class HierModel(TorchRLModule, ValueFunctionAPI):
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = PPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(RLModule)
    def setup(self):

        self.inp1 = SlimFC(HL_OWN, 150, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(HL_OPP*N_OBS, 150, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp3 = SlimFC(HL_FRI, 100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp4 = SlimFC(400, 300, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)

        required_output_dim = self.action_dist_cls.required_input_dim(space=self.action_space, model_config={})
        self.act_out = SlimFC(300,required_output_dim, activation_fn= None, initializer=torch.nn.init.orthogonal_)

        self.inp1_val = SlimFC(HL_OWN, 150, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2_val = SlimFC(HL_OPP*N_TEAM, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp3_val = SlimFC(HL_FRI*(N_TEAM-1), 150, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp4_val = SlimFC(500, 300, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.val_out = SlimFC(300, 1, activation_fn= None, initializer=torch.nn.init.orthogonal_)

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        i_own = batch[Columns.OBS][:, :HL_OWN]
        i_opp = batch[Columns.OBS][:, HL_OWN:HL_OWN + HL_OPP*N_OBS]
        i_fri = batch[Columns.OBS][:, HL_OWN + HL_OPP*N_TEAM:HL_OWN + HL_OPP*N_TEAM + HL_FRI]
        x = torch.cat((self.inp1(i_own), self.inp2(i_opp), self.inp3(i_fri)), dim=1)
        x = self.inp4(x)
        x = self.act_out(x)
        return {Columns.ACTION_DIST_INPUTS: x}

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        i_own = batch[Columns.OBS][:, :HL_OWN]
        i_opp = batch[Columns.OBS][:, HL_OWN:HL_OWN + HL_OPP*N_TEAM]
        i_fri = batch[Columns.OBS][:, HL_OWN + HL_OPP*N_TEAM:]
        x = torch.cat((self.inp1_val(i_own), self.inp2_val(i_opp), self.inp3_val(i_fri)), dim=1)
        x = self.inp4_val(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])  # or .squeeze(-1)
    

class HierModelAtt(TorchRLModule, ValueFunctionAPI):
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = PPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(RLModule)
    def setup(self):

        self.inp1 = SlimFC(HL_OWN, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(HL_OPP*N_OBS, 250, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp3 = SlimFC(HL_FRI, 50, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        
        self.act_att = nn.MultiheadAttention(250,2,batch_first=True)
        self.act_norm = nn.LayerNorm(250)

        required_output_dim = self.action_dist_cls.required_input_dim(space=self.action_space, model_config={})
        self.act_out = SlimFC(250,required_output_dim, activation_fn= None, initializer=torch.nn.init.orthogonal_)

        self.inp1_val = SlimFC(HL_OWN, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2_val = SlimFC(HL_OPP*N_TEAM, 300, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp3_val = SlimFC(HL_FRI*(N_TEAM-1), 100, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        
        self.val_att = nn.MultiheadAttention(300,2,batch_first=True)
        self.val_norm = nn.LayerNorm(300)
        
        self.val_out = SlimFC(300, 1, activation_fn= None, initializer=torch.nn.init.orthogonal_)

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        i_own = batch[Columns.OBS][:, :HL_OWN]
        i_opp = batch[Columns.OBS][:, HL_OWN:HL_OWN + HL_OPP*N_OBS]
        i_fri = batch[Columns.OBS][:, HL_OWN + HL_OPP*N_TEAM:HL_OWN + HL_OPP*N_TEAM + HL_FRI]
        
        x1 = self.inp1(i_own)
        x2 = self.inp2(i_opp)
        x3 = self.inp3(i_fri)
        
        x13 = torch.cat((x1, x3), dim=1)

        x_emb = torch.cat([x13.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        att, _ = self.act_att(x_emb, x_emb, x_emb, need_weights=False)
        att = self.act_norm(x_emb + att).mean(dim=1)
        x = self.act_out(att)

        return {Columns.ACTION_DIST_INPUTS: x}

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        i_own = batch[Columns.OBS][:, :HL_OWN]
        i_opp = batch[Columns.OBS][:, HL_OWN:HL_OWN + HL_OPP*N_TEAM]
        i_fri = batch[Columns.OBS][:, HL_OWN + HL_OPP*N_TEAM:]

        x1 = self.inp1_val(i_own)
        x2 = self.inp2_val(i_opp)
        x3 = self.inp3_val(i_fri)
        
        x13 = torch.cat((x1, x3), dim=1)

        x_emb = torch.cat([x13.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        att, _ = self.val_att(x_emb, x_emb, x_emb, need_weights=False)
        att = self.val_norm(x_emb + att).mean(dim=1)
        x = self.val_out(att)

        return torch.reshape(x, [-1])  # or .squeeze(-1)


class HierModelGRU(TorchRLModule, ValueFunctionAPI):
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = PPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(TorchRLModule)
    def get_initial_state(self):
        return {
            "h": np.zeros(shape=(200,), dtype=np.float32),
        }

    @override(RLModule)
    def setup(self):

        self.inp1 = SlimFC(HL_OWN, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(HL_OPP*N_OBS, 250, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp3 = SlimFC(HL_FRI, 50, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        
        self.act_gru = nn.GRU(250, 200, batch_first=True)
        self.act_norm = nn.LayerNorm(200)

        required_output_dim = self.action_dist_cls.required_input_dim(space=self.action_space, model_config={})
        self.act_out = SlimFC(200,required_output_dim, activation_fn= None, initializer=torch.nn.init.orthogonal_)

        self.inp1_val = SlimFC(HL_OWN, 150, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp2_val = SlimFC(HL_OPP*N_TEAM, 200, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp3_val = SlimFC(HL_FRI*(N_TEAM-1), 150, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.inp4_val = SlimFC(500, 300, activation_fn= nn.Tanh, initializer=torch.nn.init.orthogonal_)
        self.val_out = SlimFC(300, 1, activation_fn= None, initializer=torch.nn.init.orthogonal_)

    @override(RLModule)
    def _forward(self, batch, **kwargs):
        i_own = batch[Columns.OBS][:, :HL_OWN]
        i_opp = batch[Columns.OBS][:, HL_OWN:HL_OWN + HL_OPP*N_OBS]
        i_fri = batch[Columns.OBS][:, HL_OWN + HL_OPP*N_TEAM:HL_OWN + HL_OPP*N_TEAM + HL_FRI]
        x = torch.cat((self.inp1(i_own), self.inp2(i_opp), self.inp3(i_fri)), dim=1)
        x = self.inp4(x)
        x = self.act_out(x)
        return {Columns.ACTION_DIST_INPUTS: x}

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        i_own = batch[Columns.OBS][:, :HL_OWN]
        i_opp = batch[Columns.OBS][:, HL_OWN:HL_OWN + HL_OPP*N_TEAM]
        i_fri = batch[Columns.OBS][:, HL_OWN + HL_OPP*N_TEAM:]
        x = torch.cat((self.inp1_val(i_own), self.inp2_val(i_opp), self.inp3_val(i_fri)), dim=1)
        x = self.inp4_val(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])  # or .squeeze(-1)