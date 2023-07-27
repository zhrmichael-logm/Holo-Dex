import mujoco
import mujoco.viewer
import numpy as np
import time
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from holodex.constants import *
from holodex.utils.files import get_yaml_data, get_path_in_package
from scipy.spatial.transform import Rotation as R

# a class for allegro hand simulation
class AvatarSimEnv(object):
    def __init__(self, ):
        rospy.init_node('avatar_sim_env', anonymous=True)
        self.sim_config = get_yaml_data(get_path_in_package("components/simulation/configs/allegro_env.yaml"))

        self.desired_angles = None
        self.desired_wrist_pose = None

        self.num_steps = 0
        self.wrist_pos_0, self.mocap_pos_0 = None, None

        self._create_subscriber()
        self._create_publisher()
        self._create_mujoco_model()

    def _create_subscriber(self):
        rospy.Subscriber(ALLEGRO_TELEOP_ANGLES_TOPIC, Float64MultiArray, self._callback_teleop_angles, queue_size=1)
        rospy.Subscriber(ALLEGRO_WRIST_POSE_TOPIC, Float64MultiArray, self._callback_wrist_pose, queue_size=1)

    def _create_publisher(self):
        self.joint_state_pub = rospy.Publisher(ALLEGRO_JOINT_STATE_TOPIC, JointState, queue_size=1)

    def _create_mujoco_model(self): 
        self.model = mujoco.MjModel.from_xml_path(get_path_in_package(f"components/simulation/xml/{self.sim_config['scene_name']}.xml"))
        self.data = mujoco.MjData(self.model)
        self.timestep = self.sim_config['time_step']
    
    def _callback_teleop_angles(self, teleop_angles):
        self.desired_angles = np.array(list(teleop_angles.data)).reshape(16, )
    
    def _callback_wrist_pose(self, wrist_pose):
        self.desired_wrist_pose = np.array(list(wrist_pose.data)).reshape(7, )

        # self.data.mocap_pos[0] = np.array(list(wrist_pose.data)[:3])
        # self.data.mocap_quat[0] = np.array(list(wrist_pose.data)[3:])

    def _get_joint_state(self):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        # joint_state.name = self.model.joint_names
        joint_state.position = self.data.qpos
        joint_state.velocity = self.data.qvel
        joint_state.effort = self.data.qfrc_inverse
        return joint_state
    
    def _publish_joint_state(self):
        joint_state = self._get_joint_state()
        self.joint_state_pub.publish(joint_state)

    def _set_desired_angles(self):
        if self.desired_angles is not None:
            self.data.ctrl[:] = self.desired_angles
        else:
            self.data.ctrl[:] = 0.0

    def _set_desired_wrist_pose(self):
        if self.desired_wrist_pose is not None:
            wrist_pos = self.desired_wrist_pose[:3].copy()
            wrist_quat = self.desired_wrist_pose[3:].copy()
            
            # rotate wrist pose to align with mujoco world frame
            r_M_V = R.from_quat([0.701, 0, 0, 0.701])
            pos = r_M_V.apply(wrist_pos)
            rot = (r_M_V * R.from_quat(wrist_quat)).as_quat()
            # convert x,y,z,w to w,x,y,z
            rot = np.array([rot[3], rot[0], rot[1], rot[2]])
            rospy.loginfo(f"number of steps: {self.num_steps}")
            # save initial wrist position
            if self.num_steps == 0:
                self.wrist_pos_0 = pos.copy()
                self.mocap_pos_0 = np.array(self.data.mocap_pos[0])

            # set mocap position
            self.data.mocap_pos[0] = self.mocap_pos_0 + (pos - self.wrist_pos_0)
            self.data.mocap_quat[0] = rot
            
            self.num_steps += 1


    def _step(self): 
        self._publish_joint_state()
        self._set_desired_angles()
        self._set_desired_wrist_pose()
        # self.model.step(self.data)
        mujoco.mj_step(self.model, self.data)
        


    def simulate(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < 30000 and not rospy.is_shutdown():
                step_start = time.time()

                self._step()

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                # time_until_next_step = model.opt.timestep - (time.time() - step_start)
                time_until_next_step = self.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    avatar_sim = AvatarSimEnv()
    avatar_sim.simulate()
