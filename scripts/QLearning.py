#!/usr/bin/env python

# COM760 Group 30 - Collapsed School Rescue Robot
# QLearning.py - Q-Learning based autonomous navigation
# Robot learns to navigate school building and find survivors
#
# State space: 5 laser zones (32 possible states)
# Actions: forward, turn left, turn right, backward
# Rewards: +100 survivor found, +100 goal reached,
#          -100 collision, +5 closer, -1 per step

import rospy
import math
import numpy as np
import random
import pickle
import os
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf import transformations
from com760cw2_com760group30.msg import RLAction, RLReward, SurvivorDetected
from com760cw2_com760group30.srv import (
    MineRescueSetQLStatus,
    MineRescueSetQLStatusResponse)

class QLearning:

    # Actions: forward, turn left, turn right, backward
    ACTIONS = {
        0: ('forward',     0.3,  0.0),
        1: ('turn_left',   0.0,  0.5),
        2: ('turn_right',  0.0, -0.5),
        3: ('backward',   -0.2,  0.0),
    }
    N_ACTIONS = 4
    N_ZONES   = 5
    N_STATES  = 32  # 2^5

    def __init__(self):
        rospy.init_node('q_learning')

        self.active = False

        # Q-table
        self.q_table = np.zeros((self.N_STATES, self.N_ACTIONS))

        # Hyperparameters
        self.alpha         = 0.5
        self.gamma         = 0.9
        self.epsilon       = 0.9
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.995
        self.max_episodes  = 500

        # Training stats
        self.episode      = 0
        self.step_count   = 0
        self.max_steps    = 300
        self.total_reward = 0.0

        # Goal - emergency base
        self.goal = Point()
        self.goal.x = 8.0
        self.goal.y = 0.0

        # Survivor locations
        self.survivors = [
            {'id': 1, 'x': -8.0, 'y':  3.0, 'found': False},
            {'id': 2, 'x':  0.0, 'y': -3.0, 'found': False},
            {'id': 3, 'x':  7.0, 'y':  3.0, 'found': False},
        ]
        self.survivors_found = 0

        # Robot state
        self.position        = Point()
        self.yaw             = 0.0
        self.prev_dist_goal  = float('inf')
        self.laser_zones     = [0] * self.N_ZONES
        self.obs_threshold   = 0.5

        # Q-table save path
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        self.q_table_path = os.path.join(
            pkg_dir, 'q_table_school.pkl')

        # Publishers
        self.pub_vel    = rospy.Publisher(
            '/com760group30Bot/cmd_vel', Twist, queue_size=1)
        self.pub_action = rospy.Publisher(
            '/com760group30Bot/rl_action', RLAction, queue_size=1)
        self.pub_reward = rospy.Publisher(
            '/com760group30Bot/rl_reward', RLReward, queue_size=1)

        # Subscribers
        self.sub_laser = rospy.Subscriber(
            '/com760group30Bot/laser/scan',
            LaserScan, self.callback_laser)
        self.sub_odom = rospy.Subscriber(
            '/com760group30Bot/odom',
            Odometry, self.callback_odom)

        # Service
        self.srv = rospy.Service(
            'q_learning_switch',
            MineRescueSetQLStatus,
            self.handle_switch)

        self.load_q_table()

        rospy.loginfo('='*50)
        rospy.loginfo('[QLearning] School rescue Q-Learning ready!')
        rospy.loginfo('[QLearning] Call q_learning_switch to start.')
        rospy.loginfo('='*50)

        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.active:
                self.run_step()
            rate.sleep()

    def handle_switch(self, req):
        self.active = req.flag
        if req.flag:
            self.alpha        = req.learning_rate if req.learning_rate > 0 else 0.5
            self.gamma        = req.discount      if req.discount > 0      else 0.9
            self.epsilon      = req.epsilon       if req.epsilon > 0       else 0.9
            self.max_episodes = req.max_episodes  if req.max_episodes > 0  else 500
            self.episode      = 0
            # Reset survivors
            for s in self.survivors:
                s['found'] = False
            self.survivors_found = 0
            rospy.loginfo('='*50)
            rospy.loginfo('[QLearning] Training started!')
            rospy.loginfo(
                '[QLearning] alpha=%.2f gamma=%.2f '
                'epsilon=%.2f episodes=%d',
                self.alpha, self.gamma,
                self.epsilon, self.max_episodes)
            rospy.loginfo('='*50)
            return MineRescueSetQLStatusResponse(
                success=True,
                message='Q-Learning started!')
        else:
            self.stop_robot()
            self.save_q_table()
            return MineRescueSetQLStatusResponse(
                success=True,
                message='Q-Learning stopped. Q-table saved.')

    def callback_odom(self, msg):
        self.position = msg.pose.pose.position
        q = (msg.pose.pose.orientation.x,
             msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w)
        self.yaw = transformations.euler_from_quaternion(q)[2]

    def callback_laser(self, msg):
        ranges = list(msg.ranges)
        max_r  = msg.range_max
        n      = len(ranges)
        s      = max(1, int(n * 30 / 360))
        clean  = [r if (not math.isnan(r) and
                        not math.isinf(r)) else max_r
                  for r in ranges]
        raw = {
            'front':       min(min(clean[-s:]), min(clean[:s])),
            'front_left':  min(clean[s: s*3]) if s*3 < n else max_r,
            'front_right': min(clean[-(s*3):-s]) if s*3 < n else max_r,
            'left':        min(clean[s*3: n//2]) if s*3 < n//2 else max_r,
            'right':       min(clean[n//2: -(s*3)]) if s*3 < n//2 else max_r,
        }
        t = self.obs_threshold
        self.laser_zones = [
            1 if raw['front']       < t else 0,
            1 if raw['front_left']  < t else 0,
            1 if raw['front_right'] < t else 0,
            1 if raw['left']        < t else 0,
            1 if raw['right']       < t else 0,
        ]

    def get_state(self):
        return int(''.join(str(z) for z in self.laser_zones), 2)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.N_ACTIONS - 1)
        return int(np.argmax(self.q_table[state]))

    def execute_action(self, action_id):
        label, linear, angular = self.ACTIONS[action_id]
        msg = Twist()
        msg.linear.x  = linear
        msg.angular.z = angular
        self.pub_vel.publish(msg)
        action_msg = RLAction()
        action_msg.action_id     = action_id
        action_msg.linear_speed  = linear
        action_msg.angular_speed = angular
        action_msg.description   = label
        self.pub_action.publish(action_msg)

    def check_survivors(self):
        for survivor in self.survivors:
            if survivor['found']:
                continue
            dist = math.sqrt(
                (self.position.x - survivor['x'])**2 +
                (self.position.y - survivor['y'])**2)
            if dist < 1.5:
                survivor['found'] = True
                self.survivors_found += 1
                rospy.logwarn('='*50)
                rospy.logwarn(
                    '*** QL: SURVIVOR %d FOUND! ***',
                    survivor['id'])
                rospy.logwarn(
                    '*** Location: (%.1f, %.1f) ***',
                    survivor['x'], survivor['y'])
                rospy.logwarn(
                    '*** Total: %d/3 ***',
                    self.survivors_found)
                rospy.logwarn('='*50)
                return 100  # Big reward for finding survivor
        return 0

    def calculate_reward(self):
        dist         = self.distance_to_goal()
        collision    = any(z == 1 for z in self.laser_zones[:3])
        goal_reached = dist < 0.35

        # Check survivors
        survivor_reward = self.check_survivors()

        reward = -1  # step penalty

        if collision:
            reward = -100
        elif goal_reached:
            reward = 100
            rospy.logwarn('='*50)
            rospy.logwarn('*** QL: EMERGENCY BASE REACHED! ***')
            rospy.logwarn('*** SCHOOL RESCUE COMPLETE! ***')
            rospy.logwarn('='*50)
        elif survivor_reward > 0:
            reward = survivor_reward
        elif dist < self.prev_dist_goal:
            reward += 5
        else:
            reward -= 5

        self.prev_dist_goal = dist

        reward_msg = RLReward()
        reward_msg.reward            = reward
        reward_msg.collision         = collision
        reward_msg.goal_reached      = goal_reached
        reward_msg.distance_to_goal  = dist
        reward_msg.state_description = str(self.laser_zones)
        self.pub_reward.publish(reward_msg)

        return reward, collision, goal_reached

    def update_q_table(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        current   = self.q_table[state, action]
        self.q_table[state, action] = (
            current + self.alpha * (
                reward + self.gamma * best_next - current))

    def run_step(self):
        if self.episode >= self.max_episodes:
            rospy.loginfo(
                '[QLearning] Training complete! %d episodes done.',
                self.episode)
            self.active = False
            self.save_q_table()
            return

        state  = self.get_state()
        action = self.choose_action(state)
        self.execute_action(action)
        rospy.sleep(0.2)

        next_state = self.get_state()
        reward, collision, done = self.calculate_reward()
        self.update_q_table(state, action, reward, next_state)

        self.total_reward += reward
        self.step_count   += 1

        episode_done = (collision or done or
                        self.step_count >= self.max_steps)

        if episode_done:
            rospy.loginfo(
                '[QLearning] Episode %d | Steps: %d | '
                'Reward: %.1f | Epsilon: %.3f | '
                'Survivors: %d/3',
                self.episode, self.step_count,
                self.total_reward, self.epsilon,
                self.survivors_found)

            self.episode      += 1
            self.step_count    = 0
            self.total_reward  = 0.0

            # Decay exploration
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay)

            # Reset survivors for next episode
            for s in self.survivors:
                s['found'] = False
            self.survivors_found = 0

            self.stop_robot()
            rospy.sleep(1.0)
            self.prev_dist_goal = self.distance_to_goal()

    def save_q_table(self):
        try:
            with open(self.q_table_path, 'wb') as f:
                pickle.dump(self.q_table, f)
            rospy.loginfo(
                '[QLearning] Q-table saved to %s',
                self.q_table_path)
        except Exception as e:
            rospy.logwarn(
                '[QLearning] Could not save Q-table: %s', e)

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'rb') as f:
                    self.q_table = pickle.load(f)
                rospy.loginfo('[QLearning] Q-table loaded!')
            except Exception as e:
                rospy.logwarn(
                    '[QLearning] Could not load Q-table: %s', e)

    def distance_to_goal(self):
        return math.sqrt(
            (self.goal.x - self.position.x)**2 +
            (self.goal.y - self.position.y)**2)

    def stop_robot(self):
        self.pub_vel.publish(Twist())

if __name__ == '__main__':
    try:
        QLearning()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass