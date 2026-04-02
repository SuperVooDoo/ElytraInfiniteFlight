import numpy as np

def RSBoi_pitch_function(descent_time = 120, descent_angle = 33, up_angle= -50, pitch_change_rate = 0.5):
    return np.concatenate([
        np.full(int(descent_time), descent_angle),
        np.arange(up_angle, descent_angle, pitch_change_rate)
    ])

def Rectangles_pitch_function(descent_time = 160, descent_angle = 33, ascent_time = 120, ascent_angle = -33):
    return np.concatenate([
        np.full(int(descent_time), descent_angle),
        np.full(int(ascent_time), ascent_angle)
    ])

def Waves_pitch_function(descent_time = 120, descent_angle = 33, up_angle = -50, ascent_pitch_rate = 0.5, ascent_time = 120, ascent_angle = -33, down_angle= 50, descent_pitch_rate = -0.5):
    descent_phase = np.concatenate([
        np.full(int(descent_time), descent_angle),
        np.arange(up_angle, ascent_angle, ascent_pitch_rate)
    ])
    ascent_phase = np.concatenate([
        np.full(int(ascent_time), ascent_angle),
        np.arange(down_angle, descent_angle, descent_pitch_rate)
    ])
    return np.concatenate([descent_phase, ascent_phase])




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def DisplayFunction(function, N_patterns = 3):

        Y = np.tile(function(), N_patterns)

        plt.plot(Y, label=function.__name__)
        plt.xlabel("Time Steps")
        plt.ylabel("Pitch (degrees)")
        plt.title(f"Pitch Function: {function.__name__}")
        plt.legend()
        plt.show()
    
    DisplayFunction(Waves_pitch_function)