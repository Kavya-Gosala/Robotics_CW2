#!/usr/bin/env python

# COM760 Group 30 - Collapsed School Rescue Robot
# FollowWall.py - Makes robot follow walls/obstacles
# Part of Bug2 navigation algorithm

import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from com760cw2_com760group30.srv import (
    MineRescueSetBugStatus,
    MineRescueSetBugStatusResponse)

class FollowWall:

    def __init__(self):
        rospy.init_node('follow_wall')

        self.active       = False
        self.state        = 0
        self.state_labels = {
            0: 'Find wall',
            1: 'Turn',
            2: 'Follow wall'
        }

        self.linear_speed  = 0.3
        self.angular_speed = 0.5   # faster turning to clear corners
        self.turn_direction = 'left'

        self.front  = float('inf')
        self.left   = float('inf')
        self.right  = float('inf')

        self.wall_threshold  = 0.6
        self.turn_clear_dist = 0.7  # front must be this clear before resuming forward
        self.follow_dist     = 0.55
        self.last_change     = 0.0
        self.min_state_time  = 1.0

        self.pub_vel   = rospy.Publisher(
            '/com760group30Bot/cmd_vel', Twist, queue_size=1)

        self.sub_laser = rospy.Subscriber(
            '/com760group30Bot/laser/scan',
            LaserScan, self.callback_laser)

        self.srv = rospy.Service(
            'wall_follower_switch',
            MineRescueSetBugStatus,
            self.handle_switch)

        rospy.loginfo('[FollowWall] Ready and waiting...')

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.active:
                rate.sleep()
                continue
            msg = Twist()
            if self.state == 0:
                msg = self.find_wall()
            elif self.state == 1:
                msg = self.turn()
            elif self.state == 2:
                msg = self.follow_the_wall()
            else:
                rospy.logerr('[FollowWall] Unknown state!')
            self.pub_vel.publish(msg)
            rate.sleep()

    def handle_switch(self, req):
        self.active = req.flag
        if req.flag:
            if req.speed > 0:
                self.linear_speed = req.speed * 0.6
            self.turn_direction = (req.direction
                if req.direction in ['left', 'right']
                else 'left')
            self.state       = 0
            self.last_change = rospy.get_time()
            rospy.loginfo(
                '[FollowWall] Activated. Direction: %s',
                self.turn_direction)
            return MineRescueSetBugStatusResponse(
                success=True,
                message='FollowWall activated')
        else:
            self.stop_robot()
            return MineRescueSetBugStatusResponse(
                success=True,
                message='FollowWall deactivated')

    def callback_laser(self, msg):
        ranges = list(msg.ranges)
        max_r  = msg.range_max
        n      = len(ranges)
        s      = max(1, int(n * 30 / 360))
        clean  = [r if (not math.isnan(r) and
                        not math.isinf(r) and
                        r > 0.15) else max_r
                  for r in ranges]
        # angle_min=-π → index 0=rear, index n//2=forward (angle 0)
        mid = n // 2
        self.front = min(clean[mid - s: mid + s])
        self.left  = min(clean[mid + s: mid + s*3]) if mid + s*3 <= n else max_r
        self.right = min(clean[mid - s*3: mid - s]) if mid - s*3 >= 0 else max_r

    def find_wall(self):
        # Immediately react if wall is already in front (Bug2 hands off with
        # obstacle already detected — don't drive into it for min_state_time).
        if self.front < self.wall_threshold:
            self.change_state(1)
            return Twist()
        if rospy.get_time() - self.last_change > self.min_state_time:
            if self.left < 0.7 or self.right < 0.7:
                self.change_state(2)
        msg = Twist()
        msg.linear.x  = self.linear_speed
        msg.angular.z = 0.0
        return msg

    def turn(self):
        if rospy.get_time() - self.last_change > 1.5:
            if self.front > self.turn_clear_dist:
                self.change_state(2)
        msg = Twist()
        msg.linear.x  = 0.0
        msg.angular.z = (self.angular_speed
            if self.turn_direction == 'left'
            else -self.angular_speed)
        return msg

    def follow_the_wall(self):
        if rospy.get_time() - self.last_change > self.min_state_time:
            if self.front < self.wall_threshold:
                self.change_state(1)
            elif self.left > 1.5 and self.right > 1.5:
                self.change_state(0)
        msg = Twist()
        msg.linear.x = self.linear_speed
        if self.turn_direction == 'left':
            # Turned left → wall is on RIGHT. Too close → turn left (+z) to move away.
            error = self.follow_dist - self.right
            msg.angular.z = max(-0.3, min(0.3, error * 0.4))
        else:
            # Turned right → wall is on LEFT. Too close → turn right (-z) to move away.
            error = self.follow_dist - self.left
            msg.angular.z = max(-0.3, min(0.3, -error * 0.4))
        return msg

    def change_state(self, new_state):
        if self.state != new_state:
            rospy.loginfo(
                '[FollowWall] State: %s -> %s',
                self.state_labels[self.state],
                self.state_labels[new_state])
            self.state       = new_state
            self.last_change = rospy.get_time()

    def stop_robot(self):
        self.pub_vel.publish(Twist())

if __name__ == '__main__':
    try:
        FollowWall()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
        