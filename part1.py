import math
import numpy as np
import pdb
import random
from collections import deque

paddle_height = 0.2
discount = 0.5
c = 100
max_iterations = 10000

cue = np.zeros((12, 12, 3, 3, 12, 3)) # 12 x values, 12 y values, 2 x velocities, 3 y velocities, 12 paddle locations, 3 actions
num_sa = np.zeros((12, 12, 3, 3, 12, 3)) # 12 x values, 12 y values, 2 x velocities, 3 y velocities, 12 paddle locations, 3 actions

class Pong:
    def __init__(self, ball_x, ball_y, velocity_x, velocity_y, paddle_y, reward = 0, parent=None, bounce=0):
        self.reward = reward
        self.parent = parent
        self.bounce = bounce
        
        self.ball_x = ball_x
            
        self.ball_y = ball_y
            
        if velocity_x > 0 and velocity_x < 0.03:
            self.velocity_x = 0.03
        elif velocity_x < 0 and velocity_x > -0.03:
            self.velocity_x = -0.03
        else:
            self.velocity_x = velocity_x
            
        self.velocity_y = velocity_y
        
        if paddle_y < 0:
            self.paddle_y = 0
        elif paddle_y > 1 - paddle_height:
            self.paddle_y = 1 - paddle_height
        else:
            self.paddle_y = paddle_y
            
    def step(self, move):
        new_paddle_y = self.paddle_y
        new_ball_x = self.ball_x
        new_ball_y = self.ball_y
        new_velocity_x = self.velocity_x
        new_velocity_y = self.velocity_y
        new_reward = 0
        new_bounce = self.bounce
        
        # Move paddle
        if move < 0 and self.paddle_y < 1 - paddle_height:
            new_paddle_y += 0.04
        elif move > 0 and self.paddle_y > 0:
            new_paddle_y -= 0.04
                
        if new_paddle_y < 0:
            new_paddle_y = 0
        if new_paddle_y > 1:
            new_paddle_y = 1
            
        # Move x
        new_ball_x += self.velocity_x
        if new_ball_x < 0:
            new_ball_x = -new_ball_x
            new_velocity_x = -new_velocity_x
            
        # Move y
        new_ball_y += self.velocity_y
        if new_ball_y < 0:
            new_ball_y = -new_ball_y
            new_velocity_y = -new_velocity_y
        elif new_ball_y > 1:
            new_ball_y = 2 - new_ball_y
            new_velocity_y = -new_velocity_y
            
        # Check paddle bounce
        if self.ball_x < 1 and new_ball_x >= 1:
            ratio = (1 - self.ball_x) / (new_ball_x - self.ball_x)
            intersect_y = self.ball_y + ratio * (new_ball_y - self.ball_y)
            if intersect_y >= new_paddle_y and intersect_y <= new_paddle_y + paddle_height:
                new_reward = 1
                new_ball_x = 2 - new_ball_x
                randu = random.uniform(-0.015, 0.015)
                randv = random.uniform(-0.03, 0.03)
                new_velocity_x = -self.velocity_x + randu
                new_velocity_y = self.velocity_y + randv
                new_bounce += 1
            else:
                new_reward = -1
            
        # Make sure things are still valid
        if abs(new_velocity_x) < 0.03:
            if new_velocity_x < 0:
                new_velocity_x = -0.03
            else:
                new_velocity_x = 0.03
        if abs(new_velocity_x) > 1:
            if new_velocity_x < 0:
                new_velocity_x = -1
            else:
                new_velocity_x = 1
        if abs(new_velocity_y) > 1:
            if new_velocity_y < 0:
                new_velocity_y = -1
            else:
                new_velocity_y = 1
                
        return Pong(new_ball_x, new_ball_y, new_velocity_x, new_velocity_y, new_paddle_y, new_reward, self, new_bounce)
            
    def game_over(self):
        if self.ball_x > 1:
            return True
        else:
            return False
    
    def state(self):
        if self.ball_x > 1:
            (0, 0, 0, 0, 0)
        return (self.ball_x, self.ball_y, self.velocity_x, self.velocity_y, self.paddle_y)
    
    def discrete_state(self):
        if self.ball_x > 1:
            return (0, 0, 0, 0, 0)
            
        ball_x = math.floor(12 * self.ball_x)
        ball_y = math.floor(12 * self.ball_y)
        
        if ball_x == 12:
            ball_x = 11
        if ball_y == 12:
            ball_y = 11
        
        velocity_x = 0
        if self.velocity_x > 0:
            velocity_x = 1
        else:
            velocity_x = -1
            
        velocity_y = 0
        if abs(self.velocity_y) < 0.015:
            velocity_y = 0
        elif self.velocity_y > 0:
            velocity_y = 1
        elif self.velocity_y < 0:
            velocity_y = -1
            
        paddle_y = math.floor(12 * self.paddle_y / (1 - paddle_height))
        if self.paddle_y == 1 - paddle_height:
            paddle_y = 11
            
        return (int(ball_x), int(ball_y), int(velocity_x), int(velocity_y), int(paddle_y))
    
def state_str((bx, by, vx, vy, py)):
        return "(" + str(bx) + ", " + str(by) + ", " + str(vx) + ", " + str(vy) + ", " + str(py) + ")"
    
def num_lookup(s, a):
    ds = s.discrete_state()
    if ds == (0, 0, 0, 0, 0):
        return num_sa[0][0][0][0][0][0]
    return num_sa[ds[0]][ds[1]][ds[2] + 1][ds[3] + 1][ds[4]][a + 1]

def num_set(s, a, val):
    ds = s.discrete_state()
    if ds == (0, 0, 0, 0, 0):
        num_sa[0][0][0][0][0][0] = val
    else:
        num_sa[ds[0]][ds[1]][ds[2] + 1][ds[3] + 1][ds[4]][a + 1] = val
    
def num_iter(s, a):
    ds = s.discrete_state()
    if ds == (0, 0, 0, 0, 0):
        num_sa[0][0][0][0][0][0] += 1
    else:
        num_sa[ds[0]][ds[1]][ds[2] + 1][ds[3] + 1][ds[4]][a + 1] += 1

def q_lookup(s, a):
    ds = s.discrete_state()
    if ds == (0, 0, 0, 0, 0):
        return cue[0][0][0][0][0][0]
    if ds[0] >= 12 or ds[1] >= 12 or ds[4] >= 12:
        pdb.set_trace()
    return cue[ds[0]][ds[1]][ds[2] + 1][ds[3] + 1][ds[4]][a + 1]
    
def q_set(s, a, val):
    ds = s.discrete_state()
    if ds == (0, 0, 0, 0, 0):
        cue[0][0][0][0][0][0] = val
    else:
        cue[ds[0]][ds[1]][ds[2] + 1][ds[3] + 1][ds[4]][a + 1] = val
    
def exploration(s, a):
    if num_lookup(s, a) < 5:
        return float('inf')
    else:
        return q_lookup(s, a)
    
def print_q():
    for bx in range(12):
        for by in range(12):
            for vx in range(3):
                for vy in range(3):
                    for py in range(12):
                        for a in range(3):
                            if cue[bx][by][vx][vy][py][a] != 0:
                                print(state_str((bx, by, vx, vy, py)) + ": " + str(cue[bx][by][vx][vy][py][a]))
def alpha(s, a):
    # return c / (c + num_lookup(s, a))
    return 0.3
            
def play():
    initial_s = Pong(0.5, 0.5, 0.03, 0.01, 0.5 - paddle_height / 2)
    nine_bounces = 0
    
    for i in range(200):
        s = initial_s
        while not s.game_over():
            a_max = float('-inf')
            a = None
            for a_prime in [-1, 0, 1]:
                temp = q_lookup(s, a_prime)
                if temp > a_max:
                    a_max = temp
                    a = a_prime
                    
            s = s.step(a)
            
        print("Game Over!")
        print("Bounces: " + str(s.bounce))
        if s.bounce < 9:
            print("qlearn is bad and you should feel bad\n")
        else:
            nine_bounces += 1
            print("")
            
    print(nine_bounces)
            
def qlearn():
    initial_s = Pong(0.5, 0.5, 0.03, 0.01, 0.5 - paddle_height / 2)
    
    stack = deque()
    stack.append(initial_s)
    
    while stack:
        s = stack.pop()
        
        # print(state_str(s.discrete_state()))
        # print(state_str(s.state()))
        # print("")
        
        a_max = float('-inf')
        a = None
        s_next = None
        add_back = False
        for a_prime in [-1, 0, 1]:
            s_prime = s.step(a_prime)
            if num_lookup(s, a_prime) == 0 and s.reward == 0:
                add_back = True
            # if num_lookup(s, a_prime) == 10:
                # pdb.set_trace()
            temp = exploration(s, a_prime)
            if temp > a_max:
                a_max = temp
                a = a_prime
                
        if add_back:
            stack.append(s)
                
        s_next = s.step(a)
        num_iter(s, a)
        q_s_next = float('-inf')
        for a_prime in [-1, 0, 1]:
            temp = q_lookup(s_next, a_prime)
            if temp > q_s_next:
                q_s_next = temp
                
        # pdb.set_trace()
        q_new = q_lookup(s, a) + alpha(s, a) * (s.reward + discount * q_s_next - q_lookup(s, a))
        q_set(s, a, q_new)
        
        # if s.bounce >= 30:
            # print("bounce: " + str(s.bounce))
        
        if not s.game_over() and s.bounce < 9:
            stack.append(s_next)
        elif s.bounce == 9:
            break
    
    
    return

def qlearn2():
    initial_s = Pong(0.5, 0.5, 0.03, 0.01, 0.5 - paddle_height / 2)
    
    s = initial_s
    while True:
        # print(state_str(s.discrete_state()))
        # print(state_str(s.state()))
        # print("")
        
        a_max = float('-inf')
        a = None
        s_next = None
        for a_prime in [-1, 0, 1]:
            s_prime = s.step(a_prime)
            # if num_lookup(s, a_prime) == 10:
                # pdb.set_trace()
            temp = exploration(s, a_prime)
            if temp > a_max:
                a_max = temp
                a = a_prime
                
        s_next = s.step(a)
        num_iter(s, a)
        q_s_next = float('-inf')
        for a_prime in [-1, 0, 1]:
            temp = q_lookup(s_next, a_prime)
            if temp > q_s_next:
                q_s_next = temp
                
        # pdb.set_trace()
        q_new = q_lookup(s, a) + alpha(s, a) * (s.reward + discount * q_s_next - q_lookup(s, a))
        q_set(s, a, q_new)
        
        # if s.bounce >= 30:
            # print("bounce: " + str(s.bounce))
        
        if s.game_over():
            break
        
        s = s_next
    
    return

def qtrain():
    for i in range(100000):
        qlearn2()
        # num_sa = np.zeros((12, 12, 3, 3, 12, 3)) # 12 x values, 12 y values, 2 x velocities, 3 y velocities, 12 paddle locations, 3 actions
        if i % 1000 == 0:
            print(i)
            
def sarsa():
    initial_s = Pong(0.5, 0.5, 0.03, 0.01, 0.5 - paddle_height / 2)
    
    s = initial_s
    while True:
        # print(state_str(s.discrete_state()))
        # print(state_str(s.state()))
        # print("")
        
        a_max = float('-inf')
        a = None
        s_next = None
        for a_prime in [-1, 0, 1]:
            s_prime = s.step(a_prime)
            # if num_lookup(s, a_prime) == 10:
                # pdb.set_trace()
            temp = exploration(s, a_prime)
            if temp > a_max:
                a_max = temp
                a = a_prime
                
        s_next = s.step(a)
        num_iter(s, a)
        q_s_next = float('-inf')
        for a_prime in [-1, 0, 1]:
            temp = q_lookup(s_next, a_prime)
            if temp > q_s_next:
                q_s_next = temp
                
        # pdb.set_trace()
        q_new = q_lookup(s, a) + alpha(s, a) * (s.reward + discount * q_s_next - q_lookup(s, a))
        q_set(s, a, q_new)
        
        # if s.bounce >= 30:
            # print("bounce: " + str(s.bounce))
        
        if s.game_over():
            break
        
        s = s_next
    
    return
