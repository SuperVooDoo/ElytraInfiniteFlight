import math
import matplotlib.pyplot as plt
import numpy as np

class Vec2D:
    def __init__(self, x, y):
        self.x = x  # Horizontal forward speed (combines x and z)
        self.y = y  # Vertical speed (up/down)

    def add(self, dr, dy):
        return Vec2D(self.x + dr, self.y + dy)

    def multiply(self, mr, my):
        return Vec2D(self.x * mr, self.y * my)

    def __repr__(self):
        return f"(r={self.x:.3f}, y={self.y:.3f})"




def update_movement(motion: Vec2D, pitch, gravity = 0.08) -> Vec2D:
    pitch = np.radians(pitch)  # Convert pitch to radians
    cos_squared = math.cos(pitch) ** 2

    # Apply pitch-influenced gravity
    motion = motion.add(0.0, gravity * (-1.0 + cos_squared * 0.75))

    # Dive boost: converts falling speed into forward speed
    if motion.y < 0.0:
        boost = motion.y * -0.1 * cos_squared
        motion = motion.add(boost, boost)

    # Lift when looking up
    if pitch < 0.0:
        lift = motion.x * -math.sin(pitch) * 0.04
        motion = motion.add(-lift, lift * 3.2)
        motion.x = max(motion.x, 0.01)  # Prevent negative forward speed

    # Apply drag
    motion = motion.multiply(0.99, 0.98)

    return motion


def go_down(x, y, dx, dy, pitch = 30, time = 160):
    X = [x]
    Y = [y]
    for i in range(time):
        if y <= 0:
            break
        x += dx
        y += dy
        motion = update_movement(Vec2D(dx, dy), pitch)
        dx = motion.x
        dy = motion.y
        X.append(x)
        Y.append(y)
    return X, Y, dx, dy


def go_up(x, y, dx, dy, pitch):
    X = [x]
    Y = [y]

    while motion.y <= 0:
        if y <= 0:
            break
        x += dx
        y += dy
        motion = update_movement(Vec2D(dx, dy), pitch)
        dx = motion.x
        dy = motion.y
        X.append(x)
        Y.append(y)
        
    while motion.y >= 0:
        if y <= 0:
            break
        x += dx
        y += dy
        motion = update_movement(Vec2D(dx, dy), pitch)
        dx = motion.x
        dy = motion.y
        X.append(x)
        Y.append(y)

    return X, Y, dx, dy



def sim_fixed_pitch(pitch):
    position = Vec2D(0, 100.0)
    motion = Vec2D(0.01, 0.0)
    R = [position.x]
    Y = [position.y]
    while position.y >= 0.0:
        position = position.add(motion.x, motion.y)
        motion = update_movement(motion, pitch)
        R.append(position.x)
        Y.append(position.y)
        # print(f"Position: {position}, Motion: {motion}")
    return R, Y


# Simulation for alternate pitch

def sim_alternate_pitch(pitch_d, pitch_u = None, start = 100, time = 160):
    if pitch_u is None:
        pitch_u = - pitch_d
    position = Vec2D(0, start)
    peak = position.y
    motion = Vec2D(0.1, 0.0)
    X = [position.x]
    Y = [position.y]
    P = [pitch_d]

    while position.y >= 0.0 and position.x <= 10000:
        
        for i in range(time):
            if position.y <= 0:
                break
            position = position.add(motion.x, motion.y)
            motion = update_movement(motion, pitch_d)
            X.append(position.x)
            Y.append(position.y)
            P.append(pitch_d)
            i+=1
            # print(f"Position: {position}, Motion: {motion}", "Pitch:", pitch_d, "Peak:", peak, "loop 1")
            
        while motion.y <= 0:
            if position.y <= 0:
                break
            position = position.add(motion.x, motion.y)
            motion = update_movement(motion, pitch_u)
            X.append(position.x)
            Y.append(position.y)
            P.append(pitch_u)
            # print(f"Position: {position}, Motion: {motion}", "Pitch:", pitch_u, "Peak:", peak, "loop 2")
            
        while motion.y >= 0:
            if position.y <= 0:
                break
            position = position.add(motion.x, motion.y)
            motion = update_movement(motion, pitch_u)
            X.append(position.x)
            Y.append(position.y)
            P.append(pitch_u)
            # print(f"Position: {position}, Motion: {motion}", "Pitch:", pitch_u, "Peak:", peak, "loop 3")

    return X, Y, P


def simulate_pitch_list(pitch_input, start_height = 100, only_maxes = False, get_last_position = False, get_best_position = False):
    """ Simulate movement given a list of pitch values, one per time step. """

    if get_best_position:
        get_last_position = True
        only_maxes = True

    position = Vec2D(0, start_height)
    motion = Vec2D(0.01, 0.0)
    X = [position.x]
    Y = [position.y]

    for pitch in pitch_input:
        if position.y <= 0.0:
            break
        position = position.add(motion.x, motion.y)
        motion = update_movement(motion, pitch)

        X.append(position.x)
        Y.append(position.y)

    if only_maxes:
        M_X = []
        M_Y = []
        for i in range(1, len(X)-1):
            if Y[i] > Y[i-1] and Y[i] > Y[i+1]:
                M_X.append(X[i])
                M_Y.append(Y[i])
        if M_Y:
            R_X, R_Y = M_X, M_Y
        else:
            R_X, R_Y = X, Y
    else:
        R_X, R_Y = X, Y
    if get_last_position:
        return R_X[-1], R_Y[-1]
    else:
        return R_X, R_Y



n_steps = 10000

pitch = 43.4

period = 207
amplitude = 30 # degrees
pitch_sine = [amplitude * math.sin(2 * math.pi * t / period) for t in range(n_steps)]


def pitch_wiki(descent_time):
    pitch_wiki_period = [33] * descent_time + [-50 + 0.5*t for t in range(166)]
    pitch_wiki = pitch_wiki_period * (n_steps // len(pitch_wiki_period) + 1)
    return pitch_wiki



if __name__ == "__main__":
    # Example usage:


    plt.legend()
    plt.title(f"Given Pitch Simulation")
    plt.xlabel("Horizontal Distance")
    plt.ylabel("Height")
    plt.grid()
    plt.show()