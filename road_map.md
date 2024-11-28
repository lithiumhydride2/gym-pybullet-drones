# First Step

- 首先实现一个 flocking 并使用 PID 控制， 运动方式由路线点给出的 简单 demo.
- 所有的无人机都由 reynold 控制即可.

# Second Step

- 此处需要实现关于 无人机偏航角的 实现.

# drone states
```python
state = np.hstack([
            self.pos[nth_drone, :], self.quat[nth_drone, :], # 3,4
            self.rpy[nth_drone, :], self.vel[nth_drone, :],  # 3,3
            self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]
        ])
```
# TODO LIST

- [x] 需要在 flocking_aviary 中将 ctrl_freq 和 decision_freq 分离开来，因为 BaseAviary 中 step() 是按照 ctrl_freq 调用的

- [x] 性能分析 https://blog.csdn.net/weixin_40583722/article/details/121659851
```bash
python -m cProfile -o flame_of_flocking.prof flocking.py
flameprof flame_of_flocking.prof > flame_of_flocking.svg
```

- [x] 计算 gaussian_process 需要的计算压力太大了， 需要提高运算速度， 考虑使用 `GPyTorch`
    - 使用了 GPyTorch, 将运行速度提升了 5 倍
- [x] 考虑使用 方位测量 和 距离测量进行替代, 这样的 reward 似乎难以设计
- [x] 使 control_by_RL_mask 的无人机无法接受迁移控制指令
- [] 考虑如何设计 reward function， 是否把对 angular speed 的幅度限制加入 _computeTerminated()

```
Traceback (most recent call last):
  File "/home/lih/fromgit/gym-pybullet-drones/gym_pybullet_drones/src/flocking.py", line 32, in <module>
    from ..envs.FlockingAviary import FlockingAviary
ImportError: attempted relative import with no known parent package
```



# Install
- 需要自定义 pythonpath 避免 gym 使用已经注册并移动至 sitepackages 目录的环境：
```bash
发生异常: SystemExit       (note: full exception trace is shown but execution is paused at: <module>)
2
  File "/home/lih/fromgit/gym-pybullet-drones/gym_pybullet_drones/src/flocking.py", line 313, in <module> (Current frame)
    ARGS = parser.parse_args()
SystemExit: 2

```