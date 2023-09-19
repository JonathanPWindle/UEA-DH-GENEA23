import json
import pickle
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix
import numpy as np

class PoseUtils:
    def __init__(self, data=None, scalar=None, offsets=None, ds='genea', device='cpu'):
        if ds == 'zeggs':
            bvh = '038_Flirty_2_mirror_x_1_0.json'
        elif ds == 'beat':
            bvh = '1_wayne_0_1_1.json'
        elif ds == 'trinity':
            bvh = 'TestSeq002.json'
        elif ds == 'genea':
            bvh = 'utils/ds_files/genea23.json'
        else:
            bvh = 'trn_2022_v0_003.json'
        self.ds = ds
        j = json.load(open(bvh, 'r'))
        root = j['skeleton']
        j_flat = self.nodes_flat(root)
        top = self.topology(j['skeleton'])

        p_index = {node['name']: i for i, node in enumerate(j_flat)}
        j_index = {}
        i = 0
        for node in j_flat:
            if node['ntype'] != 'End Site':
                j_index[node['name']] = i 
                i += 1
        self.positions = []
        self.rotations = []
        for t in top:
            self.rotations.append((None if t[0] is None else j_index[t[0]], j_index.get(t[1], None)))
            self.positions.append((None if t[0] is None else p_index[t[0]], p_index.get(t[1], None)))

        self.device = device
        if ds == 'beat':
            self.position_idxs = pickle.load(open('beat_idxs.pkl', 'rb'))
        elif ds == 'zeggs':
            self.position_idxs = pickle.load(open('zeggs_position_idxs.pkl', 'rb'))
        elif ds == 'trinity':
            self.position_idxs = pickle.load(open('trinity_idxs.pkl', 'rb'))
        elif ds == 'genea':
            self.position_idxs = pickle.load(open('utils/ds_files/genea23_idxs.pkl', 'rb'))
        else:
            self.position_idxs = pickle.load(open('position_idxs.pkl', 'rb'))

        if data is not None:
            if data.offsets is not None:
                if self.ds == 'genea':
                    self.offsets = torch.Tensor(data.offsets[str(1).zfill(2)]).to(device)
                    print("Using offsets: ", self.offsets.shape)
                else:
                    print("Using offsets: ", data.offsets.shape)
                    self.offsets = torch.Tensor(data.offsets).to(device)
            else:
                print("No offsets provided...")
                self.offsets = None
            self.scaler = data.mmscaler
        else:
            self.scaler = scalar
            self.offsets = offsets

    def nodes_flat(self, root):
        def traverse(node):
            yield node

            for child in node.get('children', []):
                yield from traverse(child)

        return [node for node in traverse(root)]

    def topology(self, root):
        def traverse(node, parent):
            yield (None if parent is None else parent['name'], node.get('name', None))

            for child in node.get('children', []):
                yield from traverse(child, node)

        return [node for node in traverse(root, None)]


    def get_postions(self, motion, offsets, root_off, eulers=False):
        """
        Return an array of all joint positions.
        """
        if eulers:
            motion = euler_angles_to_matrix(motion.reshape(motion.shape[0], -1, 3), convention='YXZ').squeeze()
        pos = []
        Ps = []

        for p, r in zip(self.positions, self.rotations):
            if p[0] is None:
                if motion.ndim == 5:
                    parent = torch.eye(4, 4, dtype=motion.dtype, device=motion.device).unsqueeze(0).repeat(motion.shape[0], motion.shape[1], 1, 1)
                else:
                    parent = torch.eye(4, 4, dtype=motion.dtype, device=motion.device).unsqueeze(0).repeat(motion.shape[0], 1, 1)
            else:
                parent = Ps[p[0]]
            
        
            if motion.ndim == 5:
                M = torch.eye(4, 4, dtype=motion.dtype, device=motion.device).unsqueeze(0).repeat(motion.shape[0], motion.shape[1], 1, 1)
                dim = motion.shape[1]
            else:
                M = torch.eye(4, 4, dtype=motion.dtype, device=motion.device).unsqueeze(0).repeat(motion.shape[0], 1, 1)
                dim = motion.shape[0]

            if self.offsets is not None:
                offs = offsets[..., p[1], :]

            if p[0] is None:
                M[..., 3, :3] += root_off
            if self.offsets is not None:
                M[..., 3, :3] += torch.stack(dim*[offs], axis=-2)
            if r[1] is not None:
                R = motion[..., r[1], :, :]
                M[..., :3, :3] = R
            P = M @ parent
            Ps.append(P)

            pos.append(P[..., 3, :3])
        
        return torch.stack(pos, -2)

    def pose_anim(self, poses, save_path):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1)
        min_x = poses[:,:,0].min()-10
        max_x = poses[:,:,0].max()+10
        min_y = poses[:,:,1].min()-10
        max_y = poses[:,:,1].max()+10
        def animate(i):
            ax.cla()
            self.show_pose(poses[i], ax=ax)
            ax.set_ylim(min_y, max_y)
            ax.set_xlim(min_x, max_x)
            plt.axis('off')
        fig.tight_layout()
        my_animation = animation.FuncAnimation(fig, animate, frames=len(poses), \
                                        interval=1, repeat=True)

        FFwriter = animation.FFMpegWriter(fps=30)

        my_animation.save(save_path, writer=FFwriter)
        plt.close(fig)

    def pose_anim_with_gt(self, poses, gt, save_path):
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        min_x2 = gt[:,:,0].min()-10
        max_x2 = gt[:,:,0].max()+10
        min_y2 = gt[:,:,1].min()-10
        max_y2 = gt[:,:,1].max()+10
        min_x = poses[:,:,0].min()-10
        max_x = poses[:,:,0].max()+10
        min_y = poses[:,:,1].min()-10
        max_y = poses[:,:,1].max()+10
        def animate(i):
            ax.cla()
            self.show_pose(poses[i], ax=ax)
            ax.set_ylim(min_y, max_y)
            ax.set_xlim(min_x, max_x)
            ax2.cla()
            self.show_pose(gt[i], ax=ax2)
            ax2.set_ylim(min_y2, max_y2)
            ax2.set_xlim(min_x2, max_x2)
        
        fig.tight_layout()
        my_animation = animation.FuncAnimation(fig, animate, frames=len(poses), \
                                        interval=1, repeat=True)

        FFwriter = animation.FFMpegWriter(fps=30)

        my_animation.save(save_path, writer=FFwriter)
        plt.close(fig)


    def show_pose(self, pose, alpha=1, ax=None,):
        # plt.scatter(pose[:,0], pose[:,1], s=4)
        colourmap = plt.cm.tab10 #nipy_spectral, Set1,Paired   
        colours = [colourmap(i) for i in np.linspace(0, 1,10)]

        for i, p in enumerate(self.position_idxs):
            if i > 1:
                if ax is None:
                    plt.plot((pose[p[0], 0], pose[p[1], 0]), (pose[p[0], 1], pose[p[1], 1]), alpha=alpha, c=colours[(i+6)%10])
                else:
                    ax.plot((pose[p[0], 0], pose[p[1], 0]), (pose[p[0], 1], pose[p[1], 1]), alpha=alpha, c=colours[(i+6)%10])

    def plot_alpha_poses(self, poses, alpha=0.2, ax=None):
        for p in poses:
            self.show_pose(p, alpha, ax=ax)
    
    def plot_alpha_poses_alpha_range(self, poses, a_start = 0.01, a_end=1, ax=None):
        alphas = np.linspace(a_start, a_end, num=poses.shape[0])
        for p,alpha in zip(poses, alphas):
            self.show_pose(p, alpha, ax=ax)

    def seq_to_pos(self, seq, root_zero = False):
        if root_zero:
            seq_root_off = torch.zeros_like(seq[..., :3])
        else:
            seq_root_off = seq[..., :3]

        seq_rots = seq[..., 3:]
        
        if seq_rots.ndim == 3:
            seq_rots = seq_rots.reshape(seq_rots.shape[0], seq_rots.shape[1], -1 ,6)
        else:
            seq_rots = seq_rots.reshape(seq_rots.shape[0], -1 ,6)
        seq_mats = rotation_6d_to_matrix(seq_rots)
        seq_show = self.get_postions(seq_mats, self.offsets, seq_root_off)
        return seq_show
    
    def test_seq_to_pos(self, seq):
        seq_rots = seq[..., 3:]
        seq_root_off = seq[..., :3]
        if seq_rots.ndim == 3:
            seq_rots = seq_rots.reshape(seq_rots.shape[0], seq_rots.shape[1], -1 ,6)
        else:
            seq_rots = seq_rots.reshape(seq_rots.shape[0], -1 ,6)
        seq_mats = rotation_6d_to_matrix(seq_rots)
        if self.ds == 'zeggs':
            seq_euls = matrix_to_euler_angles(seq_mats, 'XYZ')
        elif self.ds == 'beat':
            seq_euls = matrix_to_euler_angles(seq_mats, 'ZYX')
        elif self.ds == 'trinity':
            seq_euls = matrix_to_euler_angles(seq_mats, 'YXZ')
        else:
            seq_euls = matrix_to_euler_angles(seq_mats, 'YXZ')
            
        seq_euls = seq_euls.cpu().detach().numpy()
        seq_euls = np.flip(seq_euls, axis=-1)
        seq_euls = seq_euls.reshape((seq_root_off.shape[0], -1)) * -1
        seq_euls = np.degrees(seq_euls)
        bvh_combined = np.concatenate([seq_root_off.cpu().detach().numpy(), seq_euls], axis=-1)

        seq_show = self.get_postions(seq_mats, self.offsets, seq_root_off).cpu().detach().numpy()
        return seq_show, bvh_combined

    def inverse_transform_pose(self, seq, scale_rots=False):
        if self.scaler is not None:
            # return torch.tensor(self.scaler.inverse_transform(seq.cpu().detach().numpy())).to(self.device)
            if not scale_rots:
                seq[..., :3] = torch.from_numpy(self.scaler.inverse_transform(seq[..., :3].cpu().detach().numpy())).to(self.device)
            else:
                seq = torch.from_numpy(self.scaler.inverse_transform(seq.cpu().detach().numpy())).to(self.device)

            return seq
        else:
            return seq
        
if __name__ == "__main__":
    # motion_path = "/Users/jonathanwindle/Documents/PhD/MotionDiffusion/data/zeggs/dof6/005_Neutral_4_x_1_0.dat"
    # motion_mmap = np.memmap(motion_path, dtype=np.float32, mode='r')
    # motion_mmap = motion_mmap.reshape((-1, 453))
    # print(motion_mmap.shape)
    # offsets = pickle.load(open('zeggs_offsets_zero_root.pkl', 'rb'))

    putils = PoseUtils(data=None, device=None, offsets=None, ds='genea')


    # pos = putils.seq_to_pos(torch.from_numpy(motion_mmap), root_zero=False)

    # print(pos.shape)


    # putils.pose_anim(pos[:100], 'test_root.mp4')
