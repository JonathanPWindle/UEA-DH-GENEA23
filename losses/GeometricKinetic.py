import torch

class GeometricKinetic(torch.nn.Module):
    def __init__(self, rotation=True, position=False, velocity=False, acceleration=True, rotation_weight=1, position_weight=1, velocity_weight=1,  acceleration_weight=1, kinetic_weight=1, prepend='train', device='cpu', **kwargs):
        super(GeometricKinetic,self).__init__()
        self.rotation = rotation
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.device = device
        self.mse = torch.nn.L1Loss().to(device)
        self.prepend = prepend
        self.rotation_weight = rotation_weight
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.kinetic_weight = kinetic_weight

    def forward(self, gt_rot=None, pred_rot=None, gt_pos=None, pred_pos=None, gt_root=None, pred_root=None):
        assert (gt_rot is not None) == self.rotation
        assert (pred_rot is not None) == self.rotation
        assert (gt_pos is not None) == self.position
        assert (pred_pos is not None) == self.position
        assert (gt_pos is not None) == self.velocity
        assert (pred_pos is not None) == self.velocity

        losses = {}
        total_loss = 0
        unweighted_loss = 0

        gt_root_vel = torch.diff(gt_root, dim=1)
        pred_root_vel = torch.diff(pred_root, dim=1)
        losses[self.prepend +'_root_velocity_loss'] = self.mse(gt_root_vel, pred_root_vel)
        losses[self.prepend +'_root_pos_loss'] = self.mse(gt_root, pred_root)
        losses[self.prepend + '_rotation_loss'] = self.mse(gt_rot, pred_rot)
        losses[self.prepend +'_position_loss'] = self.mse(gt_pos, pred_pos)
        gt_vel = torch.diff(gt_pos, dim=1)
        pred_vel = torch.diff(pred_pos, dim=1)
        gt_acc = torch.diff(gt_vel, dim=1)
        pred_acc = torch.diff(pred_vel, dim=1)
        gt_v_2 = gt_vel**2
        pred_v_2 = pred_vel**2
        losses[self.prepend +'_velocity_loss'] = self.mse(gt_vel, pred_vel)
        losses[self.prepend +'_kinetic_loss'] = self.mse(gt_v_2, pred_v_2)
        losses[self.prepend + '_acceleration_loss'] = self.mse(gt_acc, pred_acc)
        if self.rotation:
            unweighted_loss += losses[self.prepend + '_rotation_loss']
            total_loss +=  self.rotation_weight * losses[self.prepend + '_rotation_loss']
        if self.position:
            unweighted_loss += losses[self.prepend + '_position_loss']
            total_loss +=  self.position_weight * losses[self.prepend + '_position_loss']
            total_loss += self.position_weight * 1 * losses[self.prepend +'_root_pos_loss']
        if self.velocity:
            unweighted_loss += losses[self.prepend + '_velocity_loss']
            total_loss += self.velocity_weight * losses[self.prepend + '_velocity_loss']
            total_loss += self.velocity_weight * 1 * losses[self.prepend +'_root_velocity_loss']
            total_loss += self.kinetic_weight * losses[self.prepend +'_kinetic_loss']
        if self.acceleration:
            unweighted_loss += losses[self.prepend + '_acceleration_loss']
            total_loss += self.acceleration_weight * losses[self.prepend + '_acceleration_loss']
    
        return total_loss, losses[self.prepend + '_rotation_loss'], losses[self.prepend + '_position_loss'], losses[self.prepend + '_velocity_loss'], losses[self.prepend + '_acceleration_loss'], losses[self.prepend +'_root_velocity_loss'],losses[self.prepend +'_root_pos_loss'], losses[self.prepend +'_kinetic_loss'], unweighted_loss

if __name__ == "__main__":
    mse = GeometricMAE(rotation=True, position=True, velocity=True, rotation_weight=20)

    a = torch.randn(32, 90, 100)
    b = torch.randn(32, 90, 100)
    c = torch.randn(32, 90, 200)
    d = torch.randn(32, 90, 200)

    mse(a,b, c, d)





