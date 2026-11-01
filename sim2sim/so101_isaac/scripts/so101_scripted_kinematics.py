"""Read-only USD joint-chain FK and bounded Cartesian waypoint IK for the practice arm.

FK never writes simulation state. Actions remain the existing six absolute joint radians.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics

NAMES = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')

def transform(position, quaternion):
    matrix = np.eye(4)
    matrix[:3,:3] = Rotation.from_quat([*quaternion.GetImaginary(),quaternion.GetReal()]).as_matrix()
    matrix[:3,3] = position
    return matrix

class JointChain:
    def __init__(self, asset, base_position, base_xyzw, tip=(.002,0,-.095)):
        stage = Usd.Stage.Open(str(asset))
        joints = {p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
        self.frames = []
        self.limits = []
        for name in NAMES[:5]:
            p = joints[name]
            assert p.GetAttribute('physics:axis').Get() == 'Z'
            a = transform(p.GetAttribute('physics:localPos0').Get(),p.GetAttribute('physics:localRot0').Get())
            b = transform(p.GetAttribute('physics:localPos1').Get(),p.GetAttribute('physics:localRot1').Get())
            self.frames.append((a,np.linalg.inv(b)))
            self.limits.append(np.deg2rad([p.GetAttribute('physics:lowerLimit').Get(),p.GetAttribute('physics:upperLimit').Get()]))
        self.limits = np.asarray(self.limits)
        self.base = np.eye(4)
        self.base[:3,:3] = Rotation.from_quat(base_xyzw).as_matrix()
        self.base[:3,3] = base_position
        self.tip = np.array(tip)

    def fk(self, joints):
        result = self.base.copy()
        for q,(a,b) in zip(joints,self.frames):
            r = np.eye(4);r[:3,:3]=Rotation.from_rotvec([0,0,q]).as_matrix()
            result = result @ a @ r @ b
        return result[:3,3]+result[:3,:3]@self.tip, result[:3,:3], result[:3,3]

    def solve(self, position, seed, yaw=0.):
        target_rotation = Rotation.from_euler('z',yaw).as_matrix()
        def residual(q):
            p,r,_ = self.fk(q)
            return np.r_[p-position, .08*Rotation.from_matrix(target_rotation@r.T).as_rotvec()]
        lower,upper = self.limits[:,0]+.001,self.limits[:,1]-.001
        seeds = [np.clip(seed,lower,upper),np.clip(np.deg2rad([0,-40,60,70,0]),lower,upper)]
        solutions = [least_squares(residual,q,bounds=(lower,upper),max_nfev=120,ftol=1e-9,xtol=1e-9,gtol=1e-9) for q in seeds]
        best = min(solutions,key=lambda s:np.linalg.norm(s.fun))
        p,r,_ = self.fk(best.x)
        position_error = float(np.linalg.norm(p-position))
        rotation_error = float(Rotation.from_matrix(target_rotation@r.T).magnitude())
        if position_error>.006 or rotation_error>.12:
            raise RuntimeError(f'IK unreachable: pos_error={position_error:.4f}m rot_error={rotation_error:.3f}rad target={position}')
        return best.x
