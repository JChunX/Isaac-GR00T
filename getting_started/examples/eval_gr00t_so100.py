# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SO100 Real Robot
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError
from lerobot.common.robot_devices.utils import busy_wait
import rerun as rr
import rerun.blueprint as rrb

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
import sys
import os
sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm

#################################################################################


class SO100Robot:
    def __init__(self, enable_camera=False):
        # Check if calibration directory exists
        calibration_path = "/home/jasonx/Dropbox/repos/apple_pie/lerobot/.cache/calibration/so100"
        if not os.path.exists(calibration_path):
            raise FileNotFoundError(f"Calibration directory not found at: {calibration_path}")
        self.config = So100RobotConfig(calibration_dir=calibration_path)
        self.enable_camera = enable_camera

        if not enable_camera:
            self.config.cameras = {}
        
        self.robot = make_robot_from_config(self.config)

        self.follower_arms = self.robot.follower_arms
        self.leader_arms = self.robot.leader_arms

        # Build motor index map for easier reference
        self.motor_indices = {}
        start_idx = 0
        for arm_name, arm in self.follower_arms.items():
            num_motors = len(arm.motor_names)
            self.motor_indices[arm_name] = {
                "start": start_idx,
                "end": start_idx + num_motors,
                "motor_names": arm.motor_names,
            }
            start_idx += num_motors

        print("Motor indices map:", self.motor_indices)

    @contextmanager
    def activate(self):
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        self.robot.connect()

        # Apply preset configuration to all arms
        for arm_name, arm in self.follower_arms.items():
            self.set_so100_robot_preset(arm)
            print(f"Arm {arm_name} present position:", arm.read("Present_Position"))

        self.robot.is_connected = True

        print("================> SO100 Robot is fully connected =================")


    def set_so100_robot_preset(self, motor_bus):
        # Mode=0 for Position Control
        motor_bus.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        motor_bus.write("P_Coefficient", 10) # 10
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        motor_bus.write("I_Coefficient", 0)
        motor_bus.write("D_Coefficient", 32) # 32
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        motor_bus.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        motor_bus.write("Maximum_Acceleration", 254)
        motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):

        # Get current observation state for all arms
        current_state = self.robot.capture_observation()["observation.state"]
        print("current_state shape:", current_state.shape)  # Shape indicates total number of motors
        print("observation keys:", self.robot.capture_observation().keys())
        print("current_state", current_state)
        
        # Initialize target state tensor with same shape as current state
        # Shape: [total_motors_across_all_arms]
        target_state = current_state.clone()
        current_state = torch.tensor([
            0.0000, 
            183.6914, 
            162.5098, 
            60.1953,  
            -3.2520,  
            -1.0989
        ])

        # stack along same axis for len(self.follower_arms)
        current_state = current_state.repeat(len(self.follower_arms))
        self.robot.send_action(current_state)
        time.sleep(2)
        print("----------------> SO100 Robot moved to initial pose")

    def go_home(self):
        # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        print("----------------> SO100 Robot moved to home pose")
        home_state = torch.tensor([
            0.0000, 
            183.6914, 
            162.5098, 
            100.1953,  
            -3.2520,  
            -1.0989
        ])
        num_follower_arms = len(self.robot.follower_arms)
        home_state = home_state.repeat(num_follower_arms)  # Shape: [6*num_arms]

        self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_img(self, camera_name):
        # Get image from a specific camera, defaulting to webcam
        if camera_name in self.robot.cameras:
            img = self.get_observation()[f"observation.images.{camera_name}"].data.numpy()
            # Convert bgr to rgb
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            available_cameras = list(self.robot.cameras.keys()) if hasattr(self, 'cameras') else []
            print(f"Camera {camera_name} not found. Available cameras: {available_cameras}")
            return None

    def get_all_camera_images(self):
        # Return all available camera images as a dictionary
        images = {}
        for cam_name in self.robot.cameras:
            images[cam_name] = self.get_current_img(cam_name)
        return images

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        # Enable torque on all follower arms
        if not self.robot.is_connected:
            print("Robot is not connected. Cannot enable torque.")
            return
            
        for arm_name, arm in self.follower_arms.items():
            print(f"Enabling torque on {arm_name} follower arm")
            arm.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        # Disable torque on all follower arms
        if not self.robot.is_connected:
            print("Robot is not connected. Cannot disable torque.")
            return
            
        for arm_name, arm in self.follower_arms.items():
            print(f"Disabling torque on {arm_name} follower arm")
            arm.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        try:
            if self.robot.is_connected:
                self.disable()
                self.robot.disconnect()
                self.robot.is_connected = False
                print("================> SO100 Robot disconnected")
        except Exception as e:
            print(f"Error during disconnection: {e}")

    def __del__(self):
        self.disconnect()


#################################################################################


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Pick up the fruits and place them on the plate.",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, img, state):
        obs_dict = {
            "video.webcam": img[np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        start_time = time.time()
        res = self.policy.get_action(obs_dict)
        print("Inference query time taken", time.time() - start_time)
        return res

    def sample_action(self):
        obs_dict = {
            "video.webcam": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 5)),
            "state.gripper": np.zeros((1, 1)),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)

class CustomGr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Pick up the fruits and place them on the plate.",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, imgs, state):
        obs_dict = {
            "video.webcam": imgs["webcam"][np.newaxis, :, :, :],
            "video.main": imgs["main"][np.newaxis, :, :, :],
            "video.cv": imgs["cv"][np.newaxis, :, :, :],
            "state.main_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.main_gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "state.cv_arm": state[6:11][np.newaxis, :].astype(np.float64),
            "state.cv_gripper": state[11:12][np.newaxis, :].astype(np.float64),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        start_time = time.time()
        res = self.policy.get_action(obs_dict)
        print("Inference query time taken", time.time() - start_time)
        return res

    def sample_action(self):
        obs_dict = {
            "video.webcam": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "video.main": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "video.cv": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.main_arm": np.zeros((1, 5)),
            "state.main_gripper": np.zeros((1, 1)),
            "state.cv_arm": np.zeros((1, 5)),
            "state.cv_gripper": np.zeros((1, 1)),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)


#################################################################################


def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    """
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def view_imgs(imgs):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    
    Args:
        imgs: Single image or list of images to display
    """
    if isinstance(imgs, dict):
        imgs = [img for img in imgs.values()]
        
    num_imgs = len(imgs)
    cols = min(3, num_imgs)  # Max 3 columns 
    rows = (num_imgs + cols - 1) // cols
    
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.axis("off")
        
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame

#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    default_dataset_path = os.path.expanduser("~/datasets/so100_strawberry_grape")

    rr.init("gr00t_so100", spawn=True)
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--use_policy", action="store_true"
    # )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="10.110.17.183")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=12)
    parser.add_argument("--actions_to_execute", type=int, default=350)
    args = parser.parse_args()

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ['action.main_arm', 'action.main_gripper', 'action.cv_arm', 'action.cv_gripper']

    client = CustomGr00tRobotInferenceClient(
        host=args.host,
        port=args.port,
        language_instruction="Grasp items from white bowl and place in black tray",
    )

    robot = SO100Robot(enable_camera=True)
    with robot.activate():
        for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
            imgs = robot.get_all_camera_images()
            view_imgs(imgs)
            t0 = time.time()
            state = robot.get_current_state()
            t1 = time.time()
            print("state time taken", t1 - t0)
            action = client.get_action(imgs, state)
            t2 = time.time()
            print("action time taken", t2 - t1)
            start_time = time.time()
            for i in range(ACTION_HORIZON):
                concat_action = np.concatenate(
                    [np.atleast_1d(action[key][i]) for key in MODALITY_KEYS],
                    axis=0,
                )
                assert concat_action.shape == (12,), concat_action.shape
                print("concat_action", concat_action)
                print("state", state)
                robot.set_target_state(torch.from_numpy(concat_action))
                time.sleep(0.01)

                # get the realtime image
                imgs = robot.get_all_camera_images()
                view_imgs(imgs)
                state = robot.get_current_state()
                rr.set_time_seconds("time", time.time())
                for i in range(len(concat_action)):
                    rr.log(f"gr00t_so100/joint_positions{i}", rr.Scalar(
                        state[i],
                    ))
                    rr.log(f"gr00t_so100/action{i}", rr.Scalar(
                        concat_action[i],
                    ))

                # 0.05*16 = 0.8 seconds
                print("executing action", i, "time taken", time.time() - start_time)
            print("Action chunk execution time taken", time.time() - start_time)
    # else:
    #     # Test Dataset Source https://huggingface.co/datasets/youliangtan/so100_strawberry_grape
    #     dataset = LeRobotDataset(
    #         repo_id="youliangtan/so100_strawberry_grape",
    #         root=args.dataset_path,
    #     )

    #     robot = SO100Robot(calibrate=False, enable_camera=True, camera_index=args.camera_index)
    #     with robot.activate():
    #         actions = []
    #         for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Loading actions"):
    #             action = dataset[i]["action"]
    #             img = dataset[i]["observation.images.webcam"].data.numpy()
    #             # original shape (3, 480, 640) for image data
    #             realtime_img = robot.get_current_img()

    #             img = img.transpose(1, 2, 0)
    #             view_img(img, realtime_img)
    #             actions.append(action)

    #         # plot the actions
    #         plt.plot(actions)
    #         plt.show()

    #         print("Done initial pose")

    #         # Use tqdm to create a progress bar
    #         for action in tqdm(actions, desc="Executing actions"):
    #             img = robot.get_current_img()
    #             view_img(img)

    #             robot.set_target_state(action)
    #             time.sleep(0.05)

    #         print("Done all actions")
    #         robot.go_home()
    #         print("Done home")
