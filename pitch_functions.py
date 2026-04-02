import numpy as np

def RSBoi_pitch_function(descent_time = 120, descent_angle = 33, up_angle= -50, pitch_change_rate = 0.5):
    return np.concatenate([
        np.full(int(descent_time), descent_angle),
        np.arange(up_angle, descent_angle, pitch_change_rate)
    ])

def Inverse_RSBoi_pitch_function(down_angle = 50, pitch_change_rate = -0.5, ascent_time = 120, ascent_angle = -33):
    return np.concatenate([
        np.arange(down_angle, ascent_angle, -pitch_change_rate),
        np.full(int(ascent_time), ascent_angle)
    ])

def Rectangles_pitch_function(descent_time = 160, descent_angle = 33, ascent_time = 120, ascent_angle = -33):
    return np.concatenate([
        np.full(int(descent_time), descent_angle),
        np.full(int(ascent_time), ascent_angle)
    ])