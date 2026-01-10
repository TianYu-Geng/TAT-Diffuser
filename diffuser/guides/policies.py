from collections import namedtuple
import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils

# 定义一个名为 Trajectories 的结构体，包含三个字段：动作序列，观测序列，渲染观测序列
Trajectories = namedtuple('Trajectories', 'actions observations observations_render')

class Policy:
    """
    Vanilla diffuser policy from https://github.com/jannerm/diffuser/blob/maze2d/diffuser/guides/policies.py.
    """

    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    # 对条件里的观测做归一化
    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0') # 把条件从 numpy 转成 torch.Tensor，放到 GPU（这里写死了 'cuda:0'）。
        '''
        把每个条件复制 batch_size 份：
        •	原本条件是单条轨迹的约束
        •	复制后变成 [batch_size, ...]，方便一次采样多条轨迹。
        •	这里 einops.repeat 的含义：标量/向量 d → 扩展出 repeat 这个 batch 维。
        '''
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    ## 一次扩散采样得到整条轨迹
    def __call__(self, conditions, debug=False, batch_size=1):
        conditions = self._format_conditions(conditions, batch_size) # 外部传入的条件先做归一化、转 tensor、复制成 batch。

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        sample = self.diffusion_model(conditions)
        sample = utils.to_np(sample) # sample 形状大致是 [batch_size, horizon, transition_dim]

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim] # 按约定，sample 的前 action_dim 维是动作：
        actions = self.normalizer.unnormalize(actions, 'actions') # 对动作做反归一化（从标准化空间 → 环境真实动作空间）。
        # actions = np.tanh(actions)

        ## 这就是当前一步要返回给环境的动作。
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:] # 取出 sample 里动作后面的部分，作为“归一化观测序列”。
        observations = self.normalizer.unnormalize(normed_observations, 'observations') # 再把观测做反归一化，恢复到环境坐标。

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations, observations) # 最后一个obserevations是observations_render
        return action, trajectories
        # else:
        #     return action


class TATPolicy(Policy):
    """
    Policy for TAT
    """

    def __init__(self, diffusion_model, normalizer, use_tree):
        self.use_tree = use_tree # 是否启用 TAT（否则就退化为普通 Diffuser）。
        self.tree = None # 保存当前 episode 的 TrajAggTree 实例。
        super().__init__(diffusion_model, normalizer) # 调用父类的构造函数，完成扩散模型和 normalizer 的初始化。

    # 实现__call__方法会把一个普通类实例变成可调用对象
    def __call__(self, conditions, debug=False, batch_size=1):
        if self.use_tree:
            return self.tat_call(conditions, debug, batch_size)
        else:
            return super().__call__(conditions, debug, batch_size)


    def tat_call(self, conditions, debug=False, batch_size=1):
        conditions = self._format_conditions(conditions, batch_size) # 和基类一样，先归一化 + batch 化条件。

        # 用扩散模型采样一批轨迹（多条），输出 numpy。
        sample = self.diffusion_model(conditions)
        sample = utils.to_np(sample)

        ## 取出动作部分并反归一化（虽然后面这个 TAT 实现里真正用的是状态部分）。
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
    
        state_of_sample = sample[:, :, self.action_dim:] # 只取观测部分
        planning_horizon = state_of_sample.shape[1]

        # 为渲染准备一份反归一化后的观测，用来画“原始 diffusion 采样轨迹”       
        normed_observations = sample[:, :, self.action_dim:]
        observations_render = self.normalizer.unnormalize(normed_observations, 'observations')

        # 存放最终由树选出的“树规划结果轨迹”（一条）
        plan_of_tree = []

        # 把 diffusion 的多条轨迹整合进树
        self.tree.integrate_trajectories(state_of_sample)

        # Get a plan via open-loop planning
        plan_of_tree.append(state_of_sample[0,0])
        for i in range(planning_horizon - 1):
            # Acting 
            next_sample, selected_key, _, _ = self.tree.get_next_state() # 在当前根节点下，找权重最大的子节点：
            plan_of_tree.append(next_sample)

            # 执行了一步之后，就沿着这条分支继续往下走，不再考虑其他分支——这就是树上的 open-loop rollout。
            self.tree.pruning(selected_key)

        # 对这个“树规划轨迹”做反归一化，得到环境坐标下的观测轨迹。
        plan_of_tree = np.array(plan_of_tree)[None]
        observations = self.normalizer.unnormalize(plan_of_tree, 'observations')

        trajectories = Trajectories(None, observations, observations_render)
        return None, trajectories


    def reset_tree(self, traj_agg_tree):
        if self.tree is not None:
            del self.tree
        self.tree = traj_agg_tree