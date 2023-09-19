"""
parse bvh file
https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html
Save a json file with heirachy and index to the columns of the postion matrix.
Save a position matrix with all the effectors converted to positions.
"""
import enum
import json
import pickle
from re import I
import time
import numpy as np
from pytorch3d.transforms import matrix_to_rotation_6d
import torch

def deg2rad(deg):
    """Convert degrees to radians"""
    return deg * np.pi / 180.0


def rot2M(angle, key="ZXY"):
    """Return a kx3x3 rotation matrix R.
    Note: bvh format performs rotation as: vR
    Args:
        angle (ndarray of shape `(k, 3)`) of angles in radians.
        key defines the order of rotation, default is "ZXY",
        meaning R = RyRxRz.
    Returns:
        R: k rotation matrices of shape `(k, 3, 3)`.
    """

    r = {k: v for k, v in zip(key, angle.T)}
    one = np.ones_like(r["X"])
    zero = np.zeros_like(r["X"])
    cx = np.cos(r["X"])
    cy = np.cos(r["Y"])
    cz = np.cos(r["Z"])
    sx = np.sin(r["X"])
    sy = np.sin(r["Y"])
    sz = np.sin(r["Z"])

    rx = np.stack([one, zero, zero, zero, cx, sx, zero, -sx, cx], -1)
    ry = np.stack([cy, zero, -sy, zero, one, zero, sy, zero, cy], -1)
    rz = np.stack([cz, sz, zero, -sz, cz, zero, zero, zero, one], -1)

    R = {"X": rx.reshape((-1, 3, 3)),
         "Y": ry.reshape((-1, 3, 3)),
         "Z": rz.reshape((-1, 3, 3))}

    return R[key[2]] @ R[key[1]] @ R[key[0]]


def _root(k, **kwargs):
    """Helper to parse a root node"""
    header, stack, index = kwargs['header'], kwargs['stack'], kwargs['index']
    ofs = [float(i) for i in header[k+2][1:]]
    channels = header[k+3][2:]
    chn_idx = list(range(len(channels)))
    rot_idx = [i for i, c in zip(chn_idx, channels) if 'rot' in c.lower()]
    pos_idx = [i for i, c in zip(chn_idx, channels) if 'pos' in c.lower()]
    rot_order = ''.join([s[0] for s in channels if 'rot' in s.lower()])
    node = dict(
        name=" ".join(header[k][1:]),
        ntype="ROOT",
        offset=ofs,
        channels=channels,
        parent=None,
        children=[],
        rot_order=rot_order,
        rot_idx=rot_idx,
        pos_idx=pos_idx,
    )
    stack.append(node)
    index.append((node['name'], chn_idx))
    return k+4


def _joint(k, **kwargs):
    """Helper to parse a joint node"""
    header, stack, index = kwargs['header'], kwargs['stack'], kwargs['index']
    ofs = [float(i) for i in header[k+2][1:]]
    channels = header[k+3][2:]
    next_idx = index[-1][1][-1] + 1
    chn_idx = list(range(next_idx, next_idx + len(channels)))
    rot_idx = [i for i, c in zip(chn_idx, channels) if 'rot' in c.lower()]
    pos_idx = [i for i, c in zip(chn_idx, channels) if 'pos' in c.lower()]
    rot_order = ''.join([s[0] for s in channels if 'rot' in s.lower()])
    node = dict(
        name=" ".join(header[k][1:]),
        ntype="JOINT",
        offset=ofs,
        channels=channels,
        parent=stack[-1]['name'],
        children=[],
        rot_order=rot_order,
        rot_idx=rot_idx,
        pos_idx=pos_idx,
    )
    stack[-1]['children'].append(node)
    stack.append(node)
    index.append((node['name'], chn_idx))
    return k+4


def _end(k, **kwargs):
    """Helper to parse an end node"""
    header, stack, index = kwargs['header'], kwargs['stack'], kwargs['index']
    ofs = [float(i) for i in header[k+2][1:]]
    node = dict(
        name=f"{stack[-1]['name']} End Site",
        ntype="End Site",
        offset=ofs,
        channels=None,
        parent=stack[-1]['name'],
    )
    stack[-1]['children'].append(node)
    return k+4


def _pop(k, **kwargs):
    """Helper to pop the stack if end of node."""
    stack = kwargs['stack']
    stack.pop()
    return k+1


def _pass(k, **kwargs):
    """default parse helper"""
    return k+1

curr_idx = 0
def node_matrix(node, motion, offsets=None, j_index=None):
    """get Nx4x4 rotation matrix"""
    global curr_idx

    M = np.stack([np.eye(4) for _ in motion], 0)
    if offsets is None:
        M[..., 3, :3] = np.array(node['offset'])
    else:
        offs = offsets[..., j_index[node['name']], :] #TODO change this offset indexing
        M[..., 3, :3] = np.stack(M.shape[0]*[offs], axis=0)
    curr_idx += 1
    if node['ntype'] == 'ROOT':
        M[:, 3, :3] += motion[:, node['pos_idx']]

    # End nodes do not have channels
    if 'End' in node['ntype']:
        return M

    
    # if node['pos_idx']:
    #     M[:, 3, :3] += motion[:, node['pos_idx']]

    r = deg2rad(motion[:, node['rot_idx']])
    R = rot2M(r, node['rot_order'])
    M[..., :3, :3] = R

    return M


def get_postions(motion, root, offsets=None):
    """
    Return an array of all joint positions.
    """
    pos = []
    j_flat = nodes_flat(root)
    j_index = {node['name']: i for i, node in enumerate(j_flat)}
    def traverse(node, parent=None):
        if parent is None:
            parent = np.stack([np.eye(4) for _ in motion], 0)

        P = node_matrix(node, motion, offsets, j_index) @ parent
        pos.append(P[..., 3, :3])

        if 'End' in node['ntype']:
            return

        for child in node.get('children', []):
            traverse(child, P)

    traverse(root)
    return np.stack(pos, 1)

def get_offsets(root):
    """
    Return an array of all joint positions.
    """
    offs = []

    def traverse(node, parent=None):
        offs.append(np.array(node['offset']))

        if 'End' in node['ntype']:
            return

        for child in node.get('children', []):
            traverse(child, None)

    traverse(root)
    return np.stack(offs, 0)

def read_header(fname):
    header = []
    with open(fname, 'r') as f:
        for line in f:
            s = line.strip().split()
            if not s:
                continue
            if s[0].upper() == 'FRAME':
                header.append(s)
                break
            header.append(s)
    return header


def parse_bvh(fname, remove_mean=False, offsets=None):
    kwargs = dict(header=read_header(fname), stack=[], index=[])
    k = _root(1, **kwargs)
    root = kwargs['stack'][0]
    parser = {"joint": _joint, "end": _end, "}": _pop}
    n = len(kwargs['header'])
    while k < n:
        token = kwargs['header'][k][0].lower()
        k = parser.get(token, _pass)(k, **kwargs)

    frame_time = float(kwargs['header'][-1][-1])
    motion = np.genfromtxt(fname, skip_header=n)
    if remove_mean:
        x_mean = motion[:,0].mean()
        z_mean = motion[:,2].mean()

        motion[:,0] -= x_mean
        motion[:,2] -= z_mean
    positions = get_postions(motion, root, offsets)
    jsn = dict(skeleton=root, index=kwargs['index'], frame_time=frame_time)
    return jsn, positions, motion

dof6_idx = 0
def get_6DOF_angles(motion, root):
    """
    Return an array of all joint positions.
    """
    
    dof6s = []

    def traverse(node, parent=None):
        global dof6_idx
        if 'End' in node['ntype']:
            return

        r = deg2rad(motion[:, node['rot_idx']])
        R = rot2M(r, node['rot_order'])
        dof6 = matrix_to_rotation_6d(torch.from_numpy(R))
        node['dof6_idx'] = [dof6_idx, dof6_idx+1, dof6_idx+2, dof6_idx+3, dof6_idx+4, dof6_idx+5]
        dof6_idx += 6
        dof6s.append(dof6)

        for child in node.get('children', []):
            traverse(child, None)

    traverse(root)
    return np.stack(dof6s, 1)

def get_eulers_from_motion(motion, root):
    """
    Return an array of all joint positions.
    """
    pos = []
    def traverse(node, parent=None):
        if 'End' in node['ntype']:
            return
        
        R = motion[:, node['rot_idx']]
        pos.append(R)

        for child in node.get('children', []):
            traverse(child, R)

    traverse(root)
    return np.stack(pos, 1)

def write_json(fname, data):
    with open(fname, 'w') as f:
        json.dump(data, f, indent=1)

def nodes_flat(root):
    def traverse(node):
        yield node

        for child in node.get('children', []):
            yield from traverse(child)

    return [node for node in traverse(root)]

def topology(root):
    def traverse(node, parent):
        yield (None if parent is None else parent['name'], node.get('name', None))

        for child in node.get('children', []):
            yield from traverse(child, node)

    return [node for node in traverse(root, None)]
