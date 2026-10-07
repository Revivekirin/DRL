"""Audited state contracts for ManiSkill 3.0.1; no runner task dispatch.

Adding a task requires an audited declarative layout and termination semantics.
Physical-failure tasks need a synthetic termination implementation first.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class StateTask:
    extras: tuple
    poses: tuple
    invariants: tuple = ()
    booleans: tuple = ()
    relations: tuple = ()  # (vector field, destination position, source position)
    horizon: int = 50
    physical_termination: bool = False
    supported_shifts: tuple = ()

    def descriptor(self, layout):
        expected = {'agent.qpos': 9, 'agent.qvel': 9, **dict(self.extras)}
        actual = {key: stop - start for key, (start, stop) in layout.items()}
        if actual != expected:
            raise ValueError(f'Unsupported ManiSkill state layout: expected {expected}, got {actual}')
        return dict(version='maniskill_state_v1', layout=layout,
                    quaternion_poses=list(self.poses), invariant_fields=list(self.invariants),
                    boolean_fields=list(self.booleans), relations=[list(r) for r in self.relations],
                    target='aligned_quaternion_delta_v1',
                    invariant_policy='learned_delta_with_drift_diagnostic',
                    boolean_policy='continuous_prediction_without_thresholding',
                    reward_policy='learned_gaussian_without_clipping')


TASKS = {
    'PushCube-v1': StateTask(
        (('extra.tcp_pose', 7), ('extra.goal_pos', 3), ('extra.obj_pose', 7)),
        ('extra.tcp_pose', 'extra.obj_pose'), ('extra.goal_pos',)),
    'PickCube-v1': StateTask(
        (('extra.is_grasped', 1), ('extra.tcp_pose', 7), ('extra.goal_pos', 3),
         ('extra.obj_pose', 7), ('extra.tcp_to_obj_pos', 3), ('extra.obj_to_goal_pos', 3)),
        ('extra.tcp_pose', 'extra.obj_pose'), ('extra.goal_pos',), ('extra.is_grasped',),
        (('extra.tcp_to_obj_pos', 'extra.obj_pose', 'extra.tcp_pose'),
         ('extra.obj_to_goal_pos', 'extra.goal_pos', 'extra.obj_pose'))),
    'StackCube-v1': StateTask(
        (('extra.tcp_pose', 7), ('extra.cubeA_pose', 7), ('extra.cubeB_pose', 7),
         ('extra.tcp_to_cubeA_pos', 3), ('extra.tcp_to_cubeB_pos', 3), ('extra.cubeA_to_cubeB_pos', 3)),
        ('extra.tcp_pose', 'extra.cubeA_pose', 'extra.cubeB_pose'), relations=(
            ('extra.tcp_to_cubeA_pos', 'extra.cubeA_pose', 'extra.tcp_pose'),
            ('extra.tcp_to_cubeB_pos', 'extra.cubeB_pose', 'extra.tcp_pose'),
            ('extra.cubeA_to_cubeB_pos', 'extra.cubeB_pose', 'extra.cubeA_pose'))),
}


def state_task(env_id):
    if env_id not in TASKS:
        raise ValueError(f'Unsupported ManiSkill task {env_id!r}; supported: {sorted(TASKS)}')
    task = TASKS[env_id]
    if task.physical_termination:
        raise ValueError('Physical termination requires an explicit model termination codec')
    return task


def task_contract(env):
    """Factory attaches verified metadata; old injected test environments omit it."""
    return getattr(env.unwrapped, '_drl_state_contract', {})
