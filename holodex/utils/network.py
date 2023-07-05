import rospy
import zmq
import numpy as np

from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from senseglove_shared_resources.msg import SenseGloveState
from geometry_msgs.msg import PoseWithCovarianceStamped

def create_pull_socket(HOST, PORT):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind('tcp://{}:{}'.format(HOST, PORT))
    return socket

def create_push_socket(HOST, PORT):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind('tcp://{}:{}'.format(HOST, PORT))
    return socket

def frequency_timer(frequency):
    return rospy.Rate(frequency)
    
# ROS Topic Pub/Sub classes
class FloatArrayPublisher(object):
    def __init__(self, publisher_name):
        # Initializing the publisher
        self.publisher = rospy.Publisher(publisher_name, Float64MultiArray, queue_size = 1)

    def publish(self, float_array):
        data_struct = Float64MultiArray()
        data_struct.data = float_array
        self.publisher.publish(data_struct)


class ImagePublisher(object):
    def __init__(self, publisher_name, color_image = False):
        # Initializing the publisher
        self.publisher = rospy.Publisher(publisher_name, Image, queue_size = 1)
        
        # Initializing the cv bridge
        self.bridge = CvBridge()

        # Image type
        self.color_image = color_image

    def publish(self, image):
        try:
            if self.color_image:
                image = self.bridge.cv2_to_imgmsg(image, "bgr8")
            else:
                image = self.bridge.cv2_to_imgmsg(image)
        except CvBridgeError as e:
            print(e)

        self.publisher.publish(image)


class BoolPublisher(object):
    def __init__(self, publisher_name):
        # Initializing the publisher
        self.publisher = rospy.Publisher(publisher_name, Bool, queue_size = 1)

    def publish(self, bool):
        self.publisher.publish(Bool(bool))


class ImageSubscriber(object):
    def __init__(self, subscriber_name, node_name, color = True):
        try:
            rospy.init_node('{}'.format(node_name), disable_signals = True)
        except:
            pass

        self.color_image = color

        self.image = None
        self.bridge = CvBridge()
        rospy.Subscriber(subscriber_name, Image, self._callback_image, queue_size = 1, buff_size=2**24)

    def _callback_image(self, image):
        try:
            if self.color_image:
                self.image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            else:
                self.image = self.bridge.imgmsg_to_cv2(image, "passthrough")
        except CvBridgeError as e:
            print(e)

    def get_image(self):
        return self.image


class SenseGloveStateSubsciber(object):
    def __init__(self, subscriber_name, node_name):
        try:
            rospy.init_node('{}'.format(node_name), disable_signals = True)
        except:
            pass

        self.joint_angles = np.zeros((15, 3))     # 3 angles per finger(3*5), 3 element per angles (flexion, abduction, rotation)
        self.joint_positions = np.zeros((20, 3))  # 4 joints per finger(4*5, include fingertips)
        self.joint_rotations = np.zeros((20, 4))  # 4 joints per finger(4*5, include fingertips)
        self.wrist_position = None
        self.wrist_rotation = None
        rospy.Subscriber(subscriber_name, SenseGloveState, self._callback_state, queue_size = 1, buff_size=2**16)
    
    def _callback_state(self, msg):
        for i in range(15):
            self.joint_angles[i] = np.array([msg.hand_angles[i].x, msg.hand_angles[i].y, msg.hand_angles[i].z])
        for j in range(20):
            self.joint_positions[j] = np.array([msg.joint_positions[j].x, msg.joint_positions[j].y, msg.joint_positions[j].z])
            self.joint_rotations[j] = np.array([msg.joint_rotations[j].x, msg.joint_rotations[j].y, msg.joint_rotations[j].z, msg.joint_rotations[j].w])
        self.wrist_position = np.array([msg.wrist_position[0].x, msg.wrist_position[0].y, msg.wrist_position[0].z])
        self.wrist_rotation = np.array([msg.wrist_rotation[0].x, msg.wrist_rotation[0].y, msg.wrist_rotation[0].z, msg.wrist_rotation[0].w])

    def get_hand_angles(self):
        return self.joint_angles
    
    def get_joint_poses(self):
        return self.joint_positions, self.joint_rotations
    
    def get_wrist_pose(self):
        return self.wrist_position, self.wrist_rotation
    