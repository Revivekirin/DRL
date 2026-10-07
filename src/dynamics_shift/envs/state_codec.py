"""Verify actual flattened state against its structured observation."""
import numpy as np


def observation_layout(structured, flattened, task=None):
    """Discover leaf slices and verify concatenation against the actual state tensor.

    ManiSkill 3.0.1 get_obs(unflattened=True) uses ordered nested dictionaries.
    Fail on an unknown layout instead of guessing quaternion indices.
    """
    def numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    flat = numpy(flattened)
    leaves, layout = [], {}
    offset = 0

    def visit(node, prefix=''):
        nonlocal offset
        for key, value in node.items():
            path = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                visit(value, path)
            else:
                array = numpy(value).reshape(flat.shape[0], -1)
                layout[path] = [offset, offset + array.shape[1]]
                offset += array.shape[1]
                leaves.append(array)
    visit(structured)
    np.testing.assert_allclose(np.concatenate(leaves, axis=1), flat, rtol=1e-6, atol=1e-6)
    if task is not None:
        task.descriptor(layout)
        return layout
    for key, width in [('extra.tcp_pose', 7), ('extra.obj_pose', 7), ('extra.goal_pos', 3)]:
        if key not in layout or layout[key][1] - layout[key][0] != width:
            raise ValueError(f'Unsupported PushCube observation layout: {key}')
    return layout

