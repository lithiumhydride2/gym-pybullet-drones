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

[] 需要在 flocking_aviary 中将 ctrl_freq 和 decision_freq 分离开来，因为 BaseAviary 中 step() 是按照 ctrl_freq 调用的