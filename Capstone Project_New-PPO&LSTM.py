#!/usr/bin/env python
# coding: utf-8

# # Setup

# !git clone https://github.com/jpmorganchase/abides-jpmc-public

# cd /opt/anaconda3/lib/python3.9/site-packages/abides-jpmc-public

# cd /opt/anaconda3/envs/Capstone/lib/python3.8/abides-jpmc-public

# In[1]:


import ray
from ray import tune


# In[2]:


help(ray.tune.run)


# !sh install.sh

# Install in a new environment:
# 
# conda install -n myenv pip
# 
# conda activate myenv
# 
# pip <pip_subcommand>
# 
# (Same for sh)
# 
# Resources: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

# from platform import python_version
# 
# print(python_version())

# # Usage

# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


from abides_markets.configs import rmsc04
from abides_core import abides

config_state=rmsc04.build_config(seed=0,end_time='10:00:00')
end_state=abides.run(config_state)


# In[5]:


import gym
import abides_gym


# In[6]:


env=gym.make("markets-daily_investor-v0", #markets_daily_investor_environment_v0
             background_config="rmsc04"
     )

env.seed(0)


# In[7]:


state=env.reset()


# In[8]:


print(state)


# In[9]:


env.action_space


# # Step in the Simulation

# In[10]:


state,reward,done,info = env.step(0)


# In[11]:


print('State:')
print(state)
print('-------------------------------------')
print('Reward:')
print(reward)
print('-------------------------------------')
print('Done:')
print(done)
print('-------------------------------------')
print('Info:')
print(info)


# # Training an RL algorithm in ABIDES

# !pip install ray --upgrade

# In[12]:


import ray
from ray import tune

import abides_gym

from ray.tune.registry import register_env

from abides_gym.envs.markets_daily_investor_environment_v0 import (SubGymMarketsDailyInvestorEnv_v0,)


# !pip install tensorflow

# In[13]:


import tensorflow
import tensorboard


# !pip install tensorflow_probability
# import tensorflow_probability
# import numpy as np
# print(np.__version__)
# 
# !pip install --upgrade pip
# !pip install --upgrade pyproject.toml
# !python --version
# !pip uninstall numpy
# !pip install numpy==1.19.1
# 
# !pip3 install --upgrade numpy==1.19.1

# !pip3 install torch torchvision torchaudio

# execute_cell_flag=True
# 
# if execute_cell_flag:
#     ray.shutdown()
#     ray.init()
#     
#     register_env("markets-daily_investor-v0",
#                  lambda config:SubGymMarketsDailyInvestorEnv_v0(**config), )
#     
#     name_xp='dqn_market_run100_2'
#     
#     tune.run("DQN",#DQN
#              resume=True,
#              name=name_xp,
#              resume=False,
#              stop={'training_iteration':100},
#              checkpoint_at_end=True,
#              checkpoint_freq=5,
#              config={'env':"markets-daily_investor-v0",
#                      'env_config':{'background_config':'rmsc04',
#                                    'timestep_duration':"10S",
#                                    'mkt_close':"16:00:00",
#                                    'timestep_duration':"60s",
#                                    'starting_cash': 1_000_000,
#                                    'order_fixed_size': 10,
#                                    'state_history_length':4,
#                                    'market_data_buffer_length': 5,
#                                    'first_interval': "00:05:00",
#                                    'reward_mode': "dense",
#                                    'done_ratio': 0.3,
#                                    'debug_mode': True,
#                                    #'execution_window':'04:00:00',
#                                    #.... Here we just use the default values
#                                    },
#                      'seed':tune.grid_search([1]),
#              #'seed':tune.grid_search([1,2,3]),
#                      'num_gpus':0,
#                      'num_workers':0,
#                      'hiddens':[50,20],
#                      'gamma':1,
#                      'lr':tune.grid_search([0.001]),
#              #'lr':tune.grid_search([0.001,0.0001,0.01]),
#                      'framework':'torch',
#                      'observation_filter':'MeanStdFilter',
#              },
#              )

# In[24]:


get_ipython().system('pip install torch')


# In[34]:


from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

# The custom model that will be wrapped by an LSTM.
class MyCustomModel(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.num_outputs = int(np.product(self.obs_space.shape))
        self._last_batch_size = None

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        # Store last batch size for value_function output.
        self._last_batch_size = obs.shape[0]
        # Return 2x the obs (and empty states).
        # This will further be sent through an automatically provided
        # LSTM head (b/c we are setting use_lstm=True below).
        return obs * 2.0, []

    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))


# In[46]:


execute_cell_flag=True
from ray.rllib.models import ModelCatalog
import numpy as np

if execute_cell_flag:
    ray.shutdown()
    ray.init()
    
    register_env("markets-daily_investor-v0",
                 lambda config:SubGymMarketsDailyInvestorEnv_v0(**config), )
    
    name_xp='ppo_lstm_market_run100_1'
    # Register the above custom model.
    ModelCatalog.register_custom_model("my_torch_model", MyCustomModel)
    tune.run("PPO",#DQN
             name=name_xp,
             resume=False,
             stop={'training_iteration':100},
             checkpoint_at_end=True,
             checkpoint_freq=5,
             
             
             
             config={'env':"markets-daily_investor-v0",
                     'env_config':{'background_config':'rmsc04',
                                   'timestep_duration':"10S",
                                   'mkt_close':"16:00:00",
                                   'timestep_duration':"60s",
                                   'starting_cash': 1_000_000,
                                   'order_fixed_size': 10,
                                   'state_history_length':4,
                                   'market_data_buffer_length': 5,
                                   'first_interval': "00:05:00",
                                   'reward_mode': "dense",
                                   'done_ratio': 0.3,
                                   'debug_mode': True,
                                   #'execution_window':'04:00:00',
                                   #.... Here we just use the default values
                                   },
                     'model':{'use_lstm':True,
                              'lstm_cell_size':64,
                              "custom_model": "my_torch_model",
                              "custom_model_config": {}
                              },
                     'seed':tune.grid_search([1]),
             #'seed':tune.grid_search([1,2,3]),
                     #'num_gpus':0,
                     #'num_workers':0,
                     #'hiddens':[50,20],
                     'gamma':1,
                     'lr':tune.grid_search([0.001]),
             #'lr':tune.grid_search([0.001,0.0001,0.01]),
                     'framework':'torch',
                     #'observation_filter':'MeanStdFilter',
             },
             )


# !pip install tensorboard

# !pip install tensorflow

# In[1]:


get_ipython().system('tensorboard --logdir=~/ray_results/{name_xp}')


# tensorboard --logdir=~/ray_results/dpn_market_demo_3

# # Define or load a policy

# Note:
# 
# Ray Library Issue Link:https://github.com/ray-project/ray/issues/28518

# In[14]:


import numpy as np
np.random.seed(0)


# In[15]:


import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo

from ray.tune import Analysis


# In[16]:


class policyPassive:
    def __init__(self):
        self.name = 'passive'
        
    def get_action(self, state):
        return 1
        
class policyAggressive:
    def __init__(self):
        self.name = 'aggressive'
        
    def get_action(self, state):
        return 0
    
class policyRandom:
    def __init__(self):
        self.name = 'random'
        
    def get_action(self, state):
        return np.random.choice([0,1])
    
class policyRandomWithNoAction:
    def __init__(self):
        self.name = 'random_no_action'
        
    def get_action(self, state):
        return np.random.choice([0,1, 2])


# In[17]:


class policyRL:
    def __init__(self):
        self.name='r1'
        name_xp='PPO_markets-daily_investor-v0_b5e34_00000_0_lr=0.001,seed=1_2023-04-12_15-16-45'
        
        data_folder=f"~/ray_results/ppo_market_run100_1/{name_xp}"
        analysis=Analysis(data_folder)
        trail_dataframes=analysis.trial_dataframes
        trials=list(trail_dataframes.keys())
        best_trial_path=analysis.get_best_logdir(metric='episode_reward_mean',mode='max')
        best_checkpoint=analysis.get_best_checkpoint(trial=best_trial_path,mode='max')
        #best_checkpoint=complex(best_checkpoint)
        
        config=ppo.DEFAULT_CONFIG.copy()
        config['framework']='torch'
        #config['observation_filter']='MeanStdFilter'
        #config['hiddens']=[50,20]
        config['env_config']={'background_config':'rmsc04',
                              'timestep_duration':'10S',
                              'debug_mode':True
                              #Others
                             }
        self.trainer=ppo.PPOTrainer(config=config,env='markets-daily_investor-v0')
        self.trainer.restore(best_checkpoint)
    
    def get_action(self,state):
        return self.trainer.compute_action(state)      


# In[18]:


def generate_env(seed):
    env=gym.make('markets-daily_investor-v0',background_config='rmsc04',
                 timestep_duration='10S',debug_mode=True)
    env.seed(seed)
    return env


# In[19]:


from collections.abc import MutableMapping
import pandas as pd

def flatten_dict(d:MutableMapping,sep:str='.') -> MutableMapping:
    [flat_dict]=pd.json_normalize(d,sep=sep).to_dict(orient='records')
    return flat_dict


# In[20]:


def run_episode(seed=None,policy=None):
    env=generate_env(seed)
    state=env.reset()
    done=False
    episode_reward=0
    
    while not done:
        action=policy.get_action(state)
        state,reward,done,info=env.step(action)
        episode_reward += reward
    
    output=flatten_dict(info)
    output['episode_reward']=episode_reward
    output['name']=policy.name
    return output


# import p_tqdm import p_map
# import pandas as pd
# from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool
# 
# def run_N_episode(N):
#     policies=[policyAggressiove]

# In[23]:


N=20


# In[24]:


from p_tqdm import p_map
import pandas as pd 
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool


# In[25]:


def run_N_episode(N):
    """
    run in parallel N episode of testing for the different policies defined in policies list
    heads-up: does not work yet for rllib policies - pickle error
    #https://stackoverflow.com/questions/28821910/how-to-get-around-the-pickling-error-of-python-multiprocessing-without-being-in

    need to run the rllib policies with following cell (not in parralel)
    
    """
    #define policies 
    policies = [policyAggressive(), policyRandom(), policyPassive(), policyRandomWithNoAction()]
    seeds = [i for i in range(N)]
    
#     tests = [{"policy":policy, 'seed':seed} for policy in policies for seed in seeds]
    
#     def wrap_run_episode(param):
#         return run_episode(**param)
    
#     outputs = p_map(wrap_run_episode, tests)
    #outputs = Pool().map(wrap_run_episode, tests)
    outputs = []
    for policy in policies:
        for seed in seeds:
            outputs.append(run_episode(seed=seed, policy=policy))
    
    return outputs


# In[ ]:


outputs2 = run_N_episode(N)


# In[ ]:


df2 = pd.DataFrame(outputs2)
df3=df2[['name','episode_reward', 'cash', 'holdings', 'spread','marked_to_market']]
df3.to_excel('N_50_results.xlsx')


# In[ ]:


from tqdm import tqdm
# outputs=[run_episode(seed=0,policy=policyRL())]
for i in tqdm(range(N)):
    outputs2.append(run_episode(seed=i,policy=policyRL()))


# In[ ]:


df2 = pd.DataFrame(outputs2)
df3=df2[['name','episode_reward', 'cash', 'holdings', 'spread','marked_to_market']]
df3.to_excel('N_20_rl_results.xlsx')


# from tqdm import tqdm
# outputs=run_episode(seed=0,policy=policyRL())
# for i in tqdm(range(1,N)):
#     outputs.append(run_episode(seed=i,policy=policyRL()))

# # Result

# In[30]:


df2 = pd.read_excel('/Users/charlotte_jin/Desktop/N_20_rl_results.xlsx')
df2.head()


# df2 = pd.DataFrame(outputs2)
# df2.head()

# In[ ]:


df3=df2[['name','episode_reward', 'cash', 'holdings', 'spread','marked_to_market']]


# In[ ]:


df3.to_excel('N_50_results.xlsx')


# !pip install openpyxl

# In[31]:


#group by policy name and sort
df_g2 = df2.groupby(by='name').mean()

list_interest =  ['episode_reward', 'cash', 'holdings', 'spread','marked_to_market']

df_g2.sort_values(by = 'episode_reward', ascending=False)[list_interest]


# df_g = df.groupby(by='name').mean()
# 
# list_interest =  ['episode_reward', 'cash', 'holdings', 'spread','marked_to_market']
# 
# df_g.sort_values(by = 'episode_reward', ascending=False)[list_interest]

# In[86]:


df_g2['profit_percentage']=(df_g2['marked_to_market']-1000000.0)/1000000.0*100


# df_g2[['profit','profit_percentage']]

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


#distribution plots
for metric in list_interest:
    print(f'======================={metric}========================')
    sns.displot(data = df2, x=metric, hue='name', bins=10)
    plt.show()
    sns.displot(data = df2, x=metric, hue='name',  kind="kde")
    plt.show()


# # Data Simulation

# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# 
# from abides_core import abides
# from abides_core.utils import parse_logs_df, ns_date, str_to_ns, fmt_ts
# from abides_markets.configs import rmsc04
# 
# 
# config = rmsc04.build_config()
# end_state = abides.run( config )

# order_book = end_state["agents"][0].order_books["ABM"]

# order_book

# In[ ]:




