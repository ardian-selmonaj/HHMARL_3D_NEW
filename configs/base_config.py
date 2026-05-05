import argparse
import os
import datetime

# IF GPU, SET g["capturable"] = True
GPU = False
CPUS = 10

class Config(object):
    """
    Configurations for HHMARL 3D Training. 
    """
    def __init__(self,):
        parser = argparse.ArgumentParser(description='HHMARL3D Training Config')

        # training mode
        parser.add_argument('--mission', type=str, default="cannon", help='Mission to train')
        parser.add_argument('--model_name', type=str, default="f16", help='Aircraft model to train')
        parser.add_argument('--level', type=float, default=1, help='Training Level')
        parser.add_argument('--restore', type=bool, default=False, help='Restore from model')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to stored model')
        parser.add_argument('--log_name', type=str, default=None, help='Experiment Name.')
        parser.add_argument('--log_path', type=str, default=None, help='Experiment log path.')
        parser.add_argument('--config', type=dict, default=None, help='config dict for env creation')
        parser.add_argument('--hierarchy_level', type=int, default=0, help='Hierarchy Level to train, 0 for low-level, 1 for high-level.')

        # algorithm config
        parser.add_argument('--num_runners', type=int, default=2 if GPU else CPUS, help='Number of parallel samplers. Ray default 2')
        parser.add_argument('--num_runners_cpus', type=float, default=0 if GPU else 1, help='Ray default 1') #can increase (either)
        parser.add_argument('--num_runners_gpus', type=float, default=1 if GPU else 0, help='Ray default 0') #can increase (or)

        parser.add_argument('--num_learners', type=int, default=0, help='Number of parallel learners. Ray default 0') #keep 0 (main process) for stability
        parser.add_argument('--num_learners_cpus', type=float, default=0 if GPU else CPUS, help='Ray default "auto".') #can increase (either)
        parser.add_argument('--num_learners_gpus', type=float, default=1 if GPU else 0, help='Ray default 0. If num_learners=0, and learner_gpus >0, then learning on GPU.') #can increase (or)
        
        #batch 5000 for low-level (with mini 500), 3000 for high-level
        parser.add_argument('--learner_batch_size', type=int, default=4000, help='Learner batch size. Total batch size = learner_batch_size x num_learners')
        parser.add_argument('--minibatch_size', type=int, default=500, help='PPO train mini batch size')
        parser.add_argument('--num_update_iters', type=int, default=10, help='PPO SGD update iterations.')
        parser.add_argument('--train_epochs', type=int, default=40000, help='Number training epochs')
        
        self.args = parser.parse_args()
        self._set_metrics()

    def _set_metrics(self):
        if self.args.hierarchy_level == 0:
            self.args.log_name = f'L{self.args.level}_{self.args.mission}_{self.args.model_name}'
            self.args.log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', self.args.log_name)
            if self.args.restore:
                self.args.restore_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'STORED_track', 'L2_track_f16_RelAct')

            mission_config = f'{self.args.mission}/L{int(self.args.level)}_{self.args.mission}'
        
        else:
            self.args.log_name = "hier_N3"
            self.args.log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', self.args.log_name)
            if self.args.restore:
                self.args.restore_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', "hier_N3")

            mission_config = "hier/hier"

        #self.args.config = {"mission_config":mission_config, "training_level": self.args.level, "hierarchy_level":self.args.hierarchy_level} #hierarchy_level=0 -> low-level policy, =1 -> high-level commander
        self.args.config = {"mission_config":mission_config, "training_level": self.args.level+int(self.args.mission=="track"), "hierarchy_level":self.args.hierarchy_level} #hierarchy_level=0 -> low-level policy, =1 -> high-level commander
        #self.args.config = {"mission_config":mission_config, "training_level": 2, "hierarchy_level":self.args.hierarchy_level} #hierarchy_level=0 -> low-level policy, =1 -> high-level commander