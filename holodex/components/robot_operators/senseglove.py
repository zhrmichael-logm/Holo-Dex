from math import pi
import rospy
from std_msgs.msg import Float64MultiArray
from senseglove_shared_resources.msg import SenseGloveState
from .calibrators import SenseGolveThumbBoundCalibrator
from holodex.robot import AllegroKDLControl, AllegroJointControl, AllegroHand, AllegroHandSim
from holodex.utils.files import *
from holodex.utils.vec_ops import coord_in_bound
from holodex.constants import *
from copy import deepcopy as copy

class SGHapticTeleOp(object):
    def __init__(self, sim=False):
        # Initializing the ROS Node
        rospy.init_node("senseglove_haptic_teleop")

        # Storing the transformed hand coordinates
        self.hand_angles = None
        self.hand_positions = None
        self.hand_rotations = None
        self.wrist_position = None
        self.wrist_rotation = None

        # rospy.Subscriber(VR_RIGHT_TRANSFORM_COORDS_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)
        rospy.Subscriber(SG_LEFT_TRANSFORM_COORDS_TOPIC, SenseGloveState, self._callback_glove_state, queue_size=1, buff_size=2**18)

        # Initializing the solvers
        self.fingertip_solver = AllegroKDLControl()    # inverse kinematics solver in kdl package
        self.finger_joint_solver = AllegroJointControl()  # calculate joint angles from fingertip positions

        # Initializing the robot controller
        if sim:
            self.robot = AllegroHandSim()  # Simulated Allegro Hand interface
        else: self.robot = AllegroHand()  # Real Allegro Hand interface

        # Initialzing the moving average queues
        self.moving_average_queues = {
            'thumb': [],
            'index': [],
            'middle': [],
            'ring': []
        }

        # Calibrating to get the thumb bounds
        self._calibrate_bounds()

        # Getting the bounds for the allegro hand
        allegro_bounds_path = get_path_in_package('components/robot_operators/configs/allegro_vr.yaml')
        with open(allegro_bounds_path, 'r') as file:
            self.allegro_bounds = yaml.safe_load(file)

    def _calibrate_bounds(self):
        print("***************************************************************")
        print("     Starting calibration process ")
        print("***************************************************************")
        # calibrator = OculusThumbBoundCalibrator()
        calibrator = SenseGolveThumbBoundCalibrator()
        self.thumb_index_bounds, self.thumb_middle_bounds, self.thumb_ring_bounds = calibrator.get_bounds()

    def _callback_glove_state(self, msg):
        # self.hand_coords = np.array(list(coords.data)).reshape(24, 3)
        self.hand_angles = np.array([[msg.hand_angles[i].x, msg.hand_angles[i].y, msg.hand_angles[i].z] for i in range(SG_NUM_JOINTS)]).reshape(SG_NUM_JOINTS, 3)
        self.hand_positions = np.array([[msg.joint_positions[j].x, msg.joint_positions[j].y, msg.joint_positions[j].z] for j in range(SG_NUM_KEYPOINTS)]).reshape(SG_NUM_KEYPOINTS, 3) / 1000.0
        self.hand_rotations = np.array([[msg.joint_rotations[k].x, msg.joint_rotations[k].y, msg.joint_rotations[k].z, msg.joint_rotations[k].w] for k in range(SG_NUM_KEYPOINTS)]).reshape(SG_NUM_KEYPOINTS, 4)
        self.wrist_position = np.array([msg.wrist_position[0].x, msg.wrist_position[0].y, msg.wrist_position[0].z]) / 1000.0
        self.wrist_rotation = np.array([msg.wrist_rotation[0].x, msg.wrist_rotation[0].y, msg.wrist_rotation[0].z, msg.wrist_rotation[0].w])

    def _get_finger_coords(self, finger_type):
        # return np.vstack([self.hand_coords[0], self.hand_coords[OCULUS_JOINTS[finger_type]]])
        return self.hand_positions[SG_JOINTS[finger_type]]
    
    def _get_finger_angles(self, finger_type):
        return self.hand_angles[SG_JOINTS[finger_type]]

    def _get_2d_thumb_angles(self, curr_angles):
        if coord_in_bound(self.thumb_index_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_2D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_index_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['top_right'], 
                    self.allegro_bounds['thumb']['bottom_right'],
                    self.allegro_bounds['thumb']['index_bottom'],
                    self.allegro_bounds['thumb']['index_top']
                ], 
                robot_x_val = self.allegro_bounds['thumb']['x_coord'],
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        elif coord_in_bound(self.thumb_middle_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_2D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_middle_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['index_top'], 
                    self.allegro_bounds['thumb']['index_bottom'],
                    self.allegro_bounds['thumb']['middle_bottom'],
                    self.allegro_bounds['thumb']['middle_top']
                ], 
                robot_x_val = self.allegro_bounds['thumb']['x_coord'],
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        elif coord_in_bound(self.thumb_ring_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_2D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_ring_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['middle_top'], 
                    self.allegro_bounds['thumb']['middle_bottom'],
                    self.allegro_bounds['thumb']['ring_bottom'],
                    self.allegro_bounds['thumb']['ring_top']
                ], 
                robot_x_val = self.allegro_bounds['thumb']['x_coord'],
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        else:
            return curr_angles

    def _get_3d_thumb_angles(self, curr_angles):
        if coord_in_bound(self.thumb_index_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_3D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_index_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['top_right'], 
                    self.allegro_bounds['thumb']['bottom_right'],
                    self.allegro_bounds['thumb']['index_bottom'],
                    self.allegro_bounds['thumb']['index_top']
                ], 
                z_hand_bound = self.thumb_index_bounds[4], 
                x_robot_bound = [self.allegro_bounds['thumb']['index_x_bottom'], self.allegro_bounds['thumb']['index_x_top']], 
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        elif coord_in_bound(self.thumb_middle_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_3D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_middle_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['index_top'], 
                    self.allegro_bounds['thumb']['index_bottom'],
                    self.allegro_bounds['thumb']['middle_bottom'],
                    self.allegro_bounds['thumb']['middle_top']
                ], 
                z_hand_bound = self.thumb_middle_bounds[4], 
                x_robot_bound = [self.allegro_bounds['thumb']['middle_x_bottom'], self.allegro_bounds['thumb']['middle_x_top']], 
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        elif coord_in_bound(self.thumb_ring_bounds[:4], self._get_finger_coords('thumb')[-1][:2]) > -1:
            return self.fingertip_solver.thumb_motion_3D(
                hand_coordinates = self._get_finger_coords('thumb')[-1], 
                xy_hand_bounds = self.thumb_ring_bounds[:4],
                yz_robot_bounds = [
                    self.allegro_bounds['thumb']['middle_top'], 
                    self.allegro_bounds['thumb']['middle_bottom'],
                    self.allegro_bounds['thumb']['ring_bottom'],
                    self.allegro_bounds['thumb']['ring_top']
                ], 
                z_hand_bound = self.thumb_ring_bounds[4], 
                x_robot_bound = [self.allegro_bounds['thumb']['ring_x_bottom'], self.allegro_bounds['thumb']['ring_x_top']], 
                moving_avg_arr = self.moving_average_queues['thumb'], 
                curr_angles = curr_angles
            )
        else:
            return curr_angles

    def motion(self, finger_configs):
        desired_joint_angles = copy(self.robot.get_hand_position())

        # Movement for the index finger
        if finger_configs['freeze_index']:
            for idx in range(ALLEGRO_JOINTS_PER_FINGER):
                if idx > 0:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['index']] = 0.05
                else:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['index']] = 0
        elif finger_configs['finger_scale']:
            finger_angles = self._get_finger_angles('index')
            # MCP pronation
            desired_joint_angles[0 + ALLEGRO_JOINT_OFFSETS['index']] = finger_angles[0][SG_JOINT_DIRECTION['pronation']]
            # MCP flexion
            desired_joint_angles[1 + ALLEGRO_JOINT_OFFSETS['index']] = finger_angles[0][SG_JOINT_DIRECTION['flexion']] * 1.2
            # PIP flexion
            desired_joint_angles[2 + ALLEGRO_JOINT_OFFSETS['index']] = finger_angles[1][SG_JOINT_DIRECTION['flexion']] * 1.2
            # DIP flexion
            desired_joint_angles[3 + ALLEGRO_JOINT_OFFSETS['index']] = finger_angles[2][SG_JOINT_DIRECTION['flexion']] / 3.0
        else:
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type = 'index',
                finger_joint_coords = self._get_finger_coords('index'),
                curr_angles = desired_joint_angles,
                moving_avg_arr = self.moving_average_queues['index']
            )


        # Movement for the middle finger
        if finger_configs['freeze_middle']:
            for idx in range(ALLEGRO_JOINTS_PER_FINGER):
                if idx > 0:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['middle']] = 0.05
                else:
                    desired_joint_angles[idx + ALLEGRO_JOINT_OFFSETS['middle']] = 0         
        elif finger_configs['finger_scale']:
            finger_angles = self._get_finger_angles('middle')
            # MCP pronation
            desired_joint_angles[0 + ALLEGRO_JOINT_OFFSETS['middle']] = finger_angles[0][SG_JOINT_DIRECTION['pronation']]
            # MCP flexion
            desired_joint_angles[1 + ALLEGRO_JOINT_OFFSETS['middle']] = finger_angles[0][SG_JOINT_DIRECTION['flexion']] * 1.2
            # PIP flexion
            desired_joint_angles[2 + ALLEGRO_JOINT_OFFSETS['middle']] = finger_angles[1][SG_JOINT_DIRECTION['flexion']] * 1.2
            # DIP flexion
            desired_joint_angles[3 + ALLEGRO_JOINT_OFFSETS['middle']] = finger_angles[2][SG_JOINT_DIRECTION['flexion']] / 3.0
        else:
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
                finger_type = 'middle',
                finger_joint_coords = self._get_finger_coords('middle'),
                curr_angles = desired_joint_angles,
                moving_avg_arr = self.moving_average_queues['middle']
            )

        # Movement for the ring finger
        # Calculating the translatory joint angles
        if finger_configs['finger_scale']:
            finger_angles = self._get_finger_angles('ring')
            # MCP pronation
            desired_joint_angles[0 + ALLEGRO_JOINT_OFFSETS['ring']] = finger_angles[0][SG_JOINT_DIRECTION['pronation']]
            # MCP flexion
            desired_joint_angles[1 + ALLEGRO_JOINT_OFFSETS['ring']] = finger_angles[0][SG_JOINT_DIRECTION['flexion']] * 1.2
            # PIP flexion
            desired_joint_angles[2 + ALLEGRO_JOINT_OFFSETS['ring']] = finger_angles[1][SG_JOINT_DIRECTION['flexion']] * 1.2
            # DIP flexion
            desired_joint_angles[3 + ALLEGRO_JOINT_OFFSETS['ring']] = finger_angles[2][SG_JOINT_DIRECTION['flexion']] / 3.0
        else:
            desired_joint_angles = self.finger_joint_solver.calculate_finger_angles(
            finger_type = 'ring',
            finger_joint_coords = self._get_finger_coords('ring'),
            curr_angles = desired_joint_angles,
            moving_avg_arr = self.moving_average_queues['ring']
        )

        # Movement for the thumb finger - we disable 3D motion just for the thumb
        if finger_configs['thumb_scale']:
            finger_angles = self._get_finger_angles('thumb')
            # CMC flexion
            desired_joint_angles[0 + ALLEGRO_JOINT_OFFSETS['thumb']] = finger_angles[0][SG_JOINT_DIRECTION['abduction']] * 2.0 + pi/2.0
            # CMC pronation
            desired_joint_angles[1 + ALLEGRO_JOINT_OFFSETS['thumb']] = finger_angles[0][SG_JOINT_DIRECTION['pronation']] + pi/3.0
            # MCP flexion
            desired_joint_angles[2 + ALLEGRO_JOINT_OFFSETS['thumb']] = (finger_angles[0][SG_JOINT_DIRECTION['flexion']] + finger_angles[1][SG_JOINT_DIRECTION['flexion']]) * 1.0
            # PIP flexion
            desired_joint_angles[3 + ALLEGRO_JOINT_OFFSETS['thumb']] = finger_angles[2][SG_JOINT_DIRECTION['flexion']] / 1.0
        else:
            if finger_configs['three_dim']:
                desired_joint_angles = self._get_3d_thumb_angles(desired_joint_angles)
            else:
                desired_joint_angles = self._get_2d_thumb_angles(desired_joint_angles)
        
        return desired_joint_angles


    def move(self, finger_configs):
        print("\n******************************************************************************")
        print("     Controller initiated. ")
        print("******************************************************************************\n")
        print("Start controlling the robot hand using SenseGlove Haptic Nova Glove.\n")

        while not rospy.is_shutdown():
            if self.hand_angles is not None and self.robot.get_hand_position() is not None:
                # Obtaining the desired angles
                desired_joint_angles = self.motion(finger_configs)

                # Move the hand based on the desired angles
                self.robot.move(desired_joint_angles)


if __name__ == "__main__":
    teleop = SGHapticTeleOp(sim=True)
    teleop.move(finger_configs={
        'freeze_index': False,
        'freeze_middle': False,
        'freeze_ring': False,
        'freeze_thumb': False,
        'three_dim': False,
        'finger_scale': True,
        'thumb_scale': True
    })