'''
在 diffusion 每次采样出一批轨迹之后，
用一个树结构把这些轨迹在状态空间里做「聚类 + 汇总」，
把相似轨迹的公共前缀合并成树节点，
用更高权重代表“很多轨迹都走过的、有代表性的前缀”，
然后在执行时从这些高权重节点里选“最有代表性的一条前进方向”。
'''

import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def cosine_similarity(x, y):
    similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return similarity

## 权重衰减函数，表示越早的状态越重要
def get_weight(step, tree_lambda):
    return np.power(tree_lambda, step-1) 

class TreeNode(object):
    '''
    A node in the TAT.
    每个节点都会记录自己的状态、访问状态、权重和步骤（用于调试）。
    '''

    def __init__(self, parent, state, tree_lambda=0.99):
        self._parent = parent # 父节点引用
        self._children = {}  # 字典，key 是子节点索引（0,1,2,…），value 是 TreeNode。
        self._tree_lambda = tree_lambda
        self._states = [state['state']] # 所有合并到这个节点的“状态向量”列表。
        self._steps = [state['step']] # 这些状态对应的时间步索引列表。
        self._weights = [state['weight']] # 这些状态对应的权重列表（通过 get_weight 生成）
        self.node_state = np.average(np.array(self._states), axis=0, weights=np.array(self._weights)+1e-10) # 节点的“代表状态”，对 _states 做权重加权平均得到

    
    @property
    # 返回当前节点的子节点数量。用作“新子节点 key”的索引。
    def _num_children(self):
        return len(self._children)

    def expand(self, state):
        '''
        通过创建新的子节点来扩展树结构。
        '''
        self._children[self._num_children] = TreeNode(self, state, self._tree_lambda)
        return self._children[self._num_children - 1]


    def is_children(self, state, dis_threshold):
        '''
        找到最合适的节点进行转换：
        找到距离最小的子节点，如果最小距离小于阈值 dis_threshold，
        认为可以合并到这个子节点，返回 True 和对应 key；
        '''
        min_distance = 9999
        min_distance_key = None
        for key in self._children.keys():
            node_state = self._children[key].node_state
            distance = cosine_similarity(state, node_state)
            distance = 1 - distance
            if distance < min_distance:
                min_distance = distance
                min_distance_key = key
        if min_distance < dis_threshold:
            return True, min_distance_key
        else:
            return False, None
            
            
    def update_children(self, state, key):
        '''
        把一个新的状态样本（包含 state/step/weight）加入到某个子节点，
        并调用 update_node_state() 重新计算代表状态。
        '''
        self._children[key]._states.append(state['state'])
        self._children[key]._steps.append(state['step'])
        self._children[key]._weights.append(state['weight'])
        self._children[key].update_node_state()


    def update_node_state(self,):
        '''
        对 _states 按 _weights 做加权平均，更新这一节点的中心。
        '''
        self.node_state = np.average(np.array(self._states), axis=0, weights=np.array(self._weights))


    def get_value(self):
        '''
        节点的“重要程度”，用合并进来的权重总和表示。
        后面 get_next_state 会用它来选“最有代表性”的子节点。
        '''
        return np.sum(np.array(self._weights))


    def step(self, key):
        '''
        返回对应的子节点，用于在树上往下走。
        '''
        return self._children[key]


    def is_leaf(self):
        '''
        检查是否是叶节点：没有子节点 → 叶子；
        '''
        return self._children == {}

    def is_root(self):
        '''
        检查是否是根结点：没有父节点 → 根节点。
        '''
        return self._parent is None


class TrajAggTree(object):
    '''
    An implementation of Trajectory Aggregation Tree (TAT).
    '''
    def __init__(self, tree_lambda, traj_dim, action_dim=None, one_minus_alpha=0.005, start_state=None):
        self._tree_lambda = tree_lambda # 时间衰减系数 λ，控制权重随时间步衰减。
        self._distance_threshold = one_minus_alpha # 距离阈值 1-one_minus_alpha，用于判断是否可合并到已有子节点。
        self.traj_dim = traj_dim # 每一步状态向量的维度
        self.action_dim = action_dim # 如果轨迹是 [action, observation] 拼在一起，这个用于拆
        if start_state is None:
            start_state = np.zeros((traj_dim,))
        state = {'state': start_state, 'step': 0, 'weight': 1} # 根节点的初始状态，不给的话默认为全零向量。
        self._root = TreeNode(None, state, self._tree_lambda) # 树根节点，用 step=0, weight=1 初始化。


    def integrate_single_traj(self, traj, length, history_length):
        '''
        将一条单独的轨迹整合到树状结构中。
        能和已有路径“公共前缀”部分就合并；
        不相似的后半部分作为新分支挂上去。
        '''
        node = self._root

        # 合并先前的子轨迹
        for i in range(history_length, length):
            if node.is_leaf():
                break
            is_children, key = node.is_children(traj[i], self._distance_threshold)

            if is_children:
                state = {'state': traj[i], 'step': i, 'weight': get_weight(i, self._tree_lambda)} 
                node.update_children(state, key=key)
                node = node.step(key)
            else:
                # no suitable nodes for transition
                break
        
        # 扩展后一条子轨迹
        if i < length - 1:
            for j in range(i, length):
                state = {'state': traj[j], 'step': j, 'weight': get_weight(j, self._tree_lambda)}
                node = node.expand(state)


    def integrate_trajectories(self, trajectories, history_length=1):
        '''
        整合从扩散规划器中采样的一批新轨迹。
        历史长度：轨迹包含历史状态（例如，Diffuser 中包含一个历史状态）。我们将不整合历史部分。
        '''
        assert len(trajectories.shape) == 3 and trajectories.shape[-1] == self.traj_dim

        batch_size, length = trajectories.shape[0], trajectories.shape[1]

        for i in range(batch_size):
            self.integrate_single_traj(trajectories[i], length, history_length)



    def get_next_state(self,):
        '''
        Acting: select the most impactful node, which has highest weight among the child nodes.
        '''
        selected_key, node = max(self._root._children.items(), key=lambda node: node[1].get_value())
        visit_time = len(node._states)
        max_depth = np.array(node._steps).max()
        return node.node_state, selected_key, visit_time, max_depth
        

    def pruning(self, selected_key):
        '''
        修剪：修剪树木，使其与周围环境保持协调。
        '''
        self._root = self._root._children[selected_key]
        self._root._parent = None


    def forward_state(self, trajectories, action_dim=None, first_action=None):
        if action_dim is not None and action_dim != 0:
            _actions = trajectories[:, :, :self.action_dim]
            if first_action is None:
                first_action = np.zeros_like(_actions)[:, 0, :][:, None, :]
            _actions = _actions[:,:-1,:] # discard the last action
            _actions = np.concatenate([first_action, _actions], axis=1)
            _observations = trajectories[:, :, self.action_dim:]
            tree_trajectories = np.concatenate([_actions, _observations], axis=-1)
        else:
            tree_trajectories = trajectories
        return tree_trajectories
    
    def reverse_state(self, tree_trajectories, action_dim=None, last_action=None):
        if action_dim is not None and action_dim != 0:
            _actions = tree_trajectories[:, :, :self.action_dim]
            if last_action is None:
                # pad with the current last action
                last_action = _actions[:, -1, :].copy()
                last_action = last_action[:, None, :]
            _actions = _actions[:, 1:, :] # discard the first action
            _actions = np.concatenate([_actions, last_action], axis=1)
            _observations = tree_trajectories[:, :, self.action_dim:]
            trajectories = np.concatenate([_actions, _observations], axis=-1)
        else:
            trajectories = tree_trajectories
        return trajectories

    def __str__(self):
        return "TrajAggTree"