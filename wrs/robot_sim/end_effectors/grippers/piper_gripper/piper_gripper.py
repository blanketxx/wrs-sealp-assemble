import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
from wrs.grasping.grasp import *

class PiperGripper(gpi.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT,
                 name="piper_gripper"):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # flange
        # self.coupling.loc_flange_pose_list[0] = [np.zeros(3), np.eye(3)]
        self.jaw_range = np.array([.0, 0.103])
        # jlc
        Z_90_COMPENSATION = rm.rotmat_from_euler(0.0, 0.0, math.pi / 2.0)
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=2, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "base_final.stl"),
            name="piper_gripper_base",
            cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk_list[0].loc_rotmat = Z_90_COMPENSATION
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([0.3, 0.3, 0.3, 1])
        self.jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] / 2]))
        self.jlc.jnts[0].loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[0].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[0].motion_range = np.array([0.0, 0.05])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link7.stl"),
            name="piper_gripper_left_finger",
            cdmesh_type=self.cdmesh_type)
        self.jlc.jnts[0].lnk.loc_rotmat = Z_90_COMPENSATION
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1])
        self.jlc.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, np.array([0,self.jaw_range[1]/2]))
        self.jlc.jnts[1].loc_pos = np.array([0,0,0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.jnts[1].loc_motion_ax = -rm.const.y_ax
        self.jlc.jnts[1].motion_range = np.array([0, 0.05])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link8.stl"),
            name="piper_gripper_right_finger",
            cdmesh_type=self.cdmesh_type)
        self.jlc.jnts[1].lnk.loc_rotmat = Z_90_COMPENSATION
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1])
        self.jlc.finalize()
        self.loc_acting_center_pos = np.array([0, 0, 0.135])
        self.loc_acting_center_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.cdelements = (self.jlc.anchor.lnk_list[0],
                           self.jlc.jnts[0].lnk,
                           self.jlc.jnts[1].lnk)

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.pos = self._pos
        self.coupling.rotmat = self._rotmat
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def get_jaw_width(self):
        left_finger_pos = self.jlc.jnts[0].motion_value
        right_finger_pos = self.jlc.jnts[1].motion_value
        return (left_finger_pos + right_finger_pos)/1.5

    @gpi.ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        side_jawwidth = jaw_width / 2.0
        if 0 <= side_jawwidth <= self.jaw_range[1] / 2:
            self.jlc.goto_given_conf(jnt_values=[side_jawwidth, jaw_width])
        else:
            raise ValueError("The angle parameter is out of range!")

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False):
        m_col = mmc.ModelCollection(name=self.name + '_stickmodel')
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + '_meshmodel')
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_root_frame=False,
                                    toggle_flange_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.jlc.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdmesh=toggle_cdmesh,
                               toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        # oiee
        self._gen_oiee_meshmodel(m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh, toggle_frame=toggle_jnt_frames)
        return m_col
    
    def get_grasp(self, ac_pos, ac_rotmat):
        """获取当前抓取姿势"""
        return Grasp(
            ee_values=self.get_jaw_width(),  # 必须参数
            ac_pos=ac_pos,                   # 必须使用ac_pos而不是ee_pos
            ac_rotmat=ac_rotmat              # 必须使用ac_rotmat
        )

if __name__ == '__main__':
    from wrs import wd, mgm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    gripper = PiperGripper(cdmesh_type=mcm.const.CDMeshType.OBB)
    # gripper.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_euler(math.pi / 3, math.pi / 3, math.pi / 3))
    gripper.change_jaw_width(0.03)
    print(f"当前夹爪开口宽度: {gripper.get_jaw_width():.3f} m")
    gripper.gen_stickmodel().attach_to(base)
    gripper.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=False, toggle_cdprim=True,alpha=1).attach_to(base)

    base.run()