#!/usr/bin/env python

# COM760 Group 30 - Collapsed School Rescue Robot
# Bug2.py - Master coordinator implementing Bug2 algorithm
# Mission: Child 1 (-6,3) -> Teacher (0,-3) -> Child 2 (5,3) -> Emergency Base (11,0)
# Auto-starts navigation once sensors are ready.

import rospy
import math
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf import transformations
from com760cw2_com760group30.srv import (
    MineRescueSetBugStatus,
    MineRescueSetBugStatusRequest,
    MineRescueSetBugStatusResponse)
from com760cw2_com760group30.msg import SurvivorDetected

class Bug2:

    def __init__(self):
        rospy.init_node('bug2_coordinator')

        # Navigation states
        # 0 = Standing by   — waiting for sensors, then auto-starts
        # 1 = GoToPoint     — driving straight toward goal
        # 2 = FollowWall    — circumnavigating an obstacle
        # 3 = Waypoint reached — brief pause before next goal
        self.nav_state = 0
        self.nav_labels = {
            0: 'Standing by',
            1: 'GoToPoint',
            2: 'FollowWall',
            3: 'Waypoint reached'
        }

        # Robot pose
        self.position = Point()
        self.yaw      = 0.0

        # Mission waypoints — MUST match SurvivorDetector.py positions
        self.waypoints = [
            (-6.0,  3.0),   # Child 1  — NW classroom
            ( 0.0, -3.0),   # Teacher  — south corridor
            ( 5.0,  3.0),   # Child 2  — east corridor
            (11.0,  0.0),   # Emergency base — outside east wall
        ]
        self.waypoint_labels = [
            'Child 1 — NW Classroom',
            'Teacher — South Corridor',
            'Child 2 — East Corridor',
            'Emergency Base — Mission Complete',
        ]
        # Survivor waypoints (all non-last waypoints are survivors)
        self.survivor_waypoints = {0, 1, 2}
        self.current_waypoint = 0

        # Current navigation goal
        self.goal   = Point()
        self.goal.x = self.waypoints[0][0]
        self.goal.y = self.waypoints[0][1]

        # M-line parameters (Bug2 algorithm)
        self.start_position   = Point()
        self.m_line_slope     = 0.0
        self.m_line_intercept = 0.0

        # Obstacle hit-point tracking
        self.obstacle_hit_point = Point()
        self.obstacle_hit_dist  = float('inf')
        self.wall_follow_start  = 0.0   # timestamp when FollowWall began

        # Navigation parameters
        self.speed     = 0.8
        self.direction = 'left'

        # Detection thresholds
        self.obstacle_threshold   = 0.35
        self.m_line_threshold     = 0.30  # how close robot must be to M-line to switch back
        self.goal_threshold       = 1.0
        self.min_wall_follow_secs = 6.0   # must wall-follow for ≥6 s before returning
        self.m_line_progress_buf  = 0.50  # must be 0.5 m closer than hit point
        self.max_wall_follow_secs = 45.0  # escape trap: force GoToPoint after 45 s

        # Laser front distance
        self.laser_front = float('inf')

        # Sensor-ready flags (auto-start fires once both are True)
        self.got_odom  = False
        self.got_laser = False

        # ---- ROS Publishers / Subscribers ----
        self.pub_vel = rospy.Publisher(
            '/com760group30Bot/cmd_vel', Twist, queue_size=1)

        self.sub_odom = rospy.Subscriber(
            '/com760group30Bot/odom', Odometry, self.callback_odom)

        self.sub_laser = rospy.Subscriber(
            '/com760group30Bot/laser/scan', LaserScan, self.callback_laser)

        self.sub_survivor = rospy.Subscriber(
            '/com760group30Bot/survivor_detected',
            SurvivorDetected, self.callback_survivor)

        # ---- Service: external homing override ----
        self.srv_homing = rospy.Service(
            'mine_rescue_homing',
            MineRescueSetBugStatus,
            self.handle_homing_signal)

        # ---- Wait for sub-behaviour services ----
        rospy.loginfo('[Bug2] Waiting for navigation services...')
        rospy.wait_for_service('go_to_point_switch')
        rospy.wait_for_service('wall_follower_switch')

        self.client_gtp = rospy.ServiceProxy(
            'go_to_point_switch', MineRescueSetBugStatus)
        self.client_fw = rospy.ServiceProxy(
            'wall_follower_switch', MineRescueSetBugStatus)

        rospy.loginfo('=' * 55)
        rospy.loginfo('[Bug2] SYSTEM READY — Mission waypoints:')
        for i, (wp, lbl) in enumerate(
                zip(self.waypoints, self.waypoint_labels)):
            rospy.loginfo('  %d. %s -> (%.1f, %.1f)', i + 1, lbl, wp[0], wp[1])
        rospy.loginfo('[Bug2] Auto-starting once sensors are live...')
        rospy.loginfo('=' * 55)

        # ---- Main control loop ----
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.nav_state == 0:
                self.stand_by()
            elif self.nav_state == 1:
                self.bug2_go_to_point()
            elif self.nav_state == 2:
                self.bug2_follow_wall()
            elif self.nav_state == 3:
                self.waypoint_reached_behaviour()
            rate.sleep()

    # ------------------------------------------------------------------
    # State 0 — Stand by: auto-start once odometry and laser are ready
    # ------------------------------------------------------------------

    def stand_by(self):
        if self.got_odom and self.got_laser:
            rospy.loginfo('[Bug2] Sensors live. Starting mission.')
            self.start_go_to_point()

    # ------------------------------------------------------------------
    # State 1 — GoToPoint: drive directly toward goal; bail on obstacle
    # ------------------------------------------------------------------

    def bug2_go_to_point(self):
        dist = self.distance_to_goal()

        if dist < self.goal_threshold:
            self.deactivate_go_to_point()
            self.change_state(3)
            return

        if self.laser_front < self.obstacle_threshold:
            # Record where we hit the obstacle
            self.obstacle_hit_point.x = self.position.x
            self.obstacle_hit_point.y = self.position.y
            self.obstacle_hit_dist    = dist
            rospy.loginfo(
                '[Bug2] Obstacle at %.2f m. Switching to FollowWall.', dist)
            self.deactivate_go_to_point()
            self.start_follow_wall()

    # ------------------------------------------------------------------
    # State 2 — FollowWall: hug obstacle; switch back when on M-line
    # ------------------------------------------------------------------

    def bug2_follow_wall(self):
        dist = self.distance_to_goal()

        if dist < self.goal_threshold:
            self.deactivate_follow_wall()
            self.change_state(3)
            return

        elapsed = rospy.get_time() - self.wall_follow_start

        # Only check M-line return after minimum wall-follow time
        if elapsed < self.min_wall_follow_secs:
            return

        # Escape trap: if wall-following too long without M-line progress,
        # force GoToPoint from the current position so the robot doesn't loop forever.
        if elapsed > self.max_wall_follow_secs:
            rospy.logwarn(
                '[Bug2] Wall-follow timeout (%.0f s). Forcing GoToPoint.', elapsed)
            self.deactivate_follow_wall()
            self.start_position.x = self.position.x
            self.start_position.y = self.position.y
            self.compute_m_line()
            self.obstacle_hit_dist = float('inf')
            self.start_go_to_point()
            return

        # Return to GoToPoint when back on M-line AND significantly closer
        if (self.on_m_line() and
                dist < self.obstacle_hit_dist - self.m_line_progress_buf):
            rospy.loginfo(
                '[Bug2] On M-line at dist=%.2f (hit=%.2f). Back to GoToPoint.',
                dist, self.obstacle_hit_dist)
            self.deactivate_follow_wall()
            # Recompute M-line from new position
            self.start_position.x = self.position.x
            self.start_position.y = self.position.y
            self.compute_m_line()
            self.obstacle_hit_dist = float('inf')
            self.start_go_to_point()

    # ------------------------------------------------------------------
    # State 3 — Waypoint reached: advance to next goal
    # ------------------------------------------------------------------

    def waypoint_reached_behaviour(self):
        lbl = self.waypoint_labels[self.current_waypoint]
        is_last     = self.current_waypoint == len(self.waypoints) - 1
        is_survivor = self.current_waypoint in self.survivor_waypoints

        self.stop_robot()

        if is_last:
            # ---- Emergency base reached ----
            rospy.logwarn('=' * 55)
            rospy.logwarn('[Bug2] *** EMERGENCY BASE REACHED ***')
            rospy.logwarn('[Bug2] *** MISSION COMPLETE ***')
            rospy.logwarn('[Bug2] *** All survivors located and reported ***')
            rospy.logwarn('=' * 55)
            self.change_state(0)   # stand by — mission done
            return
        elif is_survivor:
            # ---- Survivor found: stop, scan, signal ----
            rospy.logwarn('=' * 55)
            rospy.logwarn('[Bug2] *** SURVIVOR LOCATION REACHED ***')
            rospy.logwarn('[Bug2] *** %s ***', lbl)
            rospy.logwarn('[Bug2] *** Position: (%.1f, %.1f) ***',
                          self.goal.x, self.goal.y)
            rospy.logwarn('[Bug2] *** Scanning for signs of life... ***')
            rospy.logwarn('=' * 55)
            rospy.sleep(3.0)   # pause at survivor location
            rospy.logwarn('[Bug2] *** SIGNAL SENT — moving to next target ***')
            rospy.logwarn('=' * 55)
        else:
            # ---- Transit waypoint: no pause, continue immediately ----
            rospy.loginfo('[Bug2] Transit waypoint %s reached.', lbl)

        self.current_waypoint += 1
        next_wp = self.waypoints[self.current_waypoint]
        self.goal.x = next_wp[0]
        self.goal.y = next_wp[1]
        rospy.loginfo('[Bug2] Next target: %s (%.1f, %.1f)',
                      self.waypoint_labels[self.current_waypoint],
                      next_wp[0], next_wp[1])

        self.start_position.x = self.position.x
        self.start_position.y = self.position.y
        self.compute_m_line()
        self.obstacle_hit_dist = float('inf')
        self.start_go_to_point()

    # ------------------------------------------------------------------
    # M-line helpers
    # ------------------------------------------------------------------

    def compute_m_line(self):
        dx = self.goal.x - self.start_position.x
        dy = self.goal.y - self.start_position.y
        if abs(dx) > 1e-6:
            self.m_line_slope     = dy / dx
            self.m_line_intercept = (self.start_position.y
                                     - self.m_line_slope * self.start_position.x)
        else:
            self.m_line_slope     = float('inf')
            self.m_line_intercept = self.start_position.x

    def on_m_line(self):
        if math.isinf(self.m_line_slope):
            return abs(self.position.x - self.m_line_intercept) < self.m_line_threshold
        perp = (abs(self.m_line_slope * self.position.x
                    - self.position.y
                    + self.m_line_intercept) /
                math.sqrt(self.m_line_slope ** 2 + 1))
        return perp < self.m_line_threshold

    def distance_to_goal(self):
        return math.sqrt(
            (self.goal.x - self.position.x) ** 2 +
            (self.goal.y - self.position.y) ** 2)

    # ------------------------------------------------------------------
    # Sub-behaviour activation
    # ------------------------------------------------------------------

    def start_go_to_point(self):
        self.start_position.x = self.position.x
        self.start_position.y = self.position.y
        self.compute_m_line()
        req = MineRescueSetBugStatusRequest()
        req.flag   = True
        req.speed  = self.speed
        req.goal_x = self.goal.x
        req.goal_y = self.goal.y
        try:
            self.client_gtp(req)
            self.change_state(1)
        except rospy.ServiceException as exc:
            rospy.logerr('[Bug2] go_to_point_switch failed: %s', exc)

    def deactivate_go_to_point(self):
        req = MineRescueSetBugStatusRequest()
        req.flag = False
        try:
            self.client_gtp(req)
        except rospy.ServiceException as exc:
            rospy.logerr('[Bug2] go_to_point_switch deactivate: %s', exc)

    def start_follow_wall(self):
        req = MineRescueSetBugStatusRequest()
        req.flag      = True
        req.speed     = self.speed
        req.direction = self.direction
        try:
            self.client_fw(req)
            self.wall_follow_start = rospy.get_time()
            self.change_state(2)
        except rospy.ServiceException as exc:
            rospy.logerr('[Bug2] wall_follower_switch failed: %s', exc)

    def deactivate_follow_wall(self):
        req = MineRescueSetBugStatusRequest()
        req.flag = False
        try:
            self.client_fw(req)
        except rospy.ServiceException as exc:
            rospy.logerr('[Bug2] wall_follower_switch deactivate: %s', exc)

    def stop_robot(self):
        self.pub_vel.publish(Twist())

    def change_state(self, new_state):
        if self.nav_state != new_state:
            rospy.loginfo('[Bug2] State: %s -> %s',
                          self.nav_labels.get(self.nav_state),
                          self.nav_labels.get(new_state))
            self.nav_state = new_state

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def callback_odom(self, msg):
        self.position = msg.pose.pose.position
        quat = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        self.yaw      = transformations.euler_from_quaternion(quat)[2]
        self.got_odom = True

    def callback_laser(self, msg):
        ranges = list(msg.ranges)
        max_r  = msg.range_max
        n      = len(ranges)
        s      = max(1, int(n * 30 / 360))   # ±30° front cone
        clean  = [r if (not math.isnan(r) and
                        not math.isinf(r) and
                        r > 0.15) else max_r
                  for r in ranges]
        # angle_min=-π → index 0=rear, index n//2=forward (angle 0)
        mid = n // 2
        self.laser_front = min(clean[mid - s: mid + s])
        self.got_laser   = True

    def callback_survivor(self, msg):
        rospy.logwarn('[Bug2] Survivor confirmed: ID=%d at (%.1f, %.1f)',
                      msg.survivor_id, msg.position_x, msg.position_y)

    # ------------------------------------------------------------------
    # External homing override service
    # ------------------------------------------------------------------

    def handle_homing_signal(self, req):
        if req.flag:
            rospy.logwarn('[Bug2] External homing signal! Jumping to emergency base.')
            self.current_waypoint = len(self.waypoints) - 1
            self.goal.x = self.waypoints[-1][0]
            self.goal.y = self.waypoints[-1][1]
            self.deactivate_follow_wall()
            self.start_go_to_point()
            return MineRescueSetBugStatusResponse(
                success=True, message='Homing to emergency base')
        return MineRescueSetBugStatusResponse(
            success=False, message='Flag not set')


if __name__ == '__main__':
    try:
        Bug2()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
