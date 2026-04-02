from distro import name
import numpy as np
import csv
import cma
import dill
import matplotlib.pyplot as plt

from ElytraMovement import *


class SolutionTemplate:

    def __init__(self, logic_pattern: callable, parameters_range: np.ndarray, initial_parameters = None):
        """
        A template for optimization problems using CMA-ES depending of a function to generate pitches 

        logic_pattern: a callable that takes in a list of parameters and returns a list of pitch values (one per time step).
        parameters_range: a 2D numpy array where each row contains the lower and upper bounds for a parameter.
        initial_parameters: a list of initial parameter values to start the optimization from. The length of this list should match the number of parameters expected by the function.
        
        """

        self.logic_pattern = logic_pattern
        self.parameters_range = np.array(parameters_range)
        self.TOTAL_TIME = 500
        self.starting_height = 100
        self.verbose = False
        self.goal = None
        self.optimized_parameters= None
        self.optimized_position = None
        
        if initial_parameters is None:
            self.initial_parameters = np.ones(len(parameters_range)) * 0.5  # Default to the middle of the range if no initial parameters are provided
        else:
            unscaled_parameters = np.array(initial_parameters)
            lower = self.parameters_range[:, 0]
            upper = self.parameters_range[:, 1]
            self.initial_parameters = (unscaled_parameters - lower) / (upper - lower)  # Scale initial parameters to [0, 1]



    def pitches(self, normalized_params):
        scaled_params = [low + (high - low) * param for (low, high), param in zip(self.parameters_range, normalized_params)]
        root_array = np.array(self.logic_pattern(*scaled_params))
        return np.resize(root_array, self.TOTAL_TIME)


    def objective(self, params, goal = 'max_height'):

        pitches = np.clip(self.pitches(params), -90, 90)
        best_position = simulate_pitch_list(pitches, get_best_position=True)

        if goal == 'max_height':
            return -best_position[1]
        elif goal == 'max_distance':
            height_penalty = max(0, 100 - best_position[1])
            return -best_position[0] + height_penalty * (self.TOTAL_TIME / 10)  # Penalize distance based on how much height is lost, scaled by total time
        else:
            raise ValueError("Invalid goal specified. Choose from 'max_height', 'max_distance', or 'best_rate_of_climb'.")
    
    def set_total_time(self, total_time):
        self.TOTAL_TIME = total_time

    def set_starting_height(self, starting_height):
        self.starting_height = starting_height

    def set_goal(self, goal):
        if goal not in ['max_height', 'max_distance']:
            raise ValueError("Invalid goal specified. Choose from 'max_height' or 'max_distance'.")
        self.goal = goal

    def set_verbose(self, verbose=None):
        if verbose is None:
            self.verbose = not self.verbose  # Toggle if no argument is provided
        else:
            self.verbose = verbose


    def optimize(self, sigma=0.2, maxiter=3000, tolx=1e-5, verbdisp=100):

        lower = np.zeros_like(self.initial_parameters)  # CMA-ES expects bounds in the scaled space
        upper = np.ones_like(self.initial_parameters)

        cma_verbose = 1 if self.verbose else -9  # CMA-ES verbosity level

        es = cma.CMAEvolutionStrategy(
            self.initial_parameters, sigma,
            {
                'bounds':    [lower, upper],
                'maxiter':   maxiter,
                'tolx':      tolx,
                'verb_disp': verbdisp,
                'verbose':  cma_verbose
            }
        )

        while not es.stop():
            solutions = es.ask()
            fitnesses = [self.objective(sol, goal=self.goal) for sol in solutions]
            es.tell(solutions, fitnesses)
            if self.verbose:
                es.disp()

        best_params  = es.result.xbest
        best_pitches = self.pitches(best_params)
        self.optimized_parameters = best_params
        self.optimized_position = simulate_pitch_list(best_pitches, start_height=self.starting_height, get_best_position=True)


    def best_optimized(self, N=10): # Run the optimization multiple times and keep the best result based on the specified goal

        best_goal = 0
        best_parameters = None
        best_position = None

        for i in range(N):

            if self.verbose:
                print(f"\nOptimization run {i+1}/{N} for goal: {self.goal}")
            self.optimize()
            if self.goal == 'max_height':
                current_goal = self.optimized_position[1]
            elif self.goal == 'max_distance':
                current_goal = self.optimized_position[0]
            else:
                raise ValueError("Invalid goal specified. Choose from 'max_height', 'max_distance', or 'best_rate_of_climb'.")

            if current_goal > best_goal:
                best_goal = current_goal
                best_parameters = self.optimized_parameters
                best_position = self.optimized_position
        
        self.optimized_parameters = best_parameters
        self.optimized_position = best_position


    def print_results(self):

        if not self.verbose:
            return

        print(f"\nOptimized parameters: {[(low + (high - low) * param) for (low, high), param in zip(self.parameters_range, self.optimized_parameters)]}")
        print(f"\nFinal position: {self.optimized_position[0]:.1f}, {self.optimized_position[1]:.1f}")
        print(f"Horizontal speed: {self.optimized_position[0] / (self.TOTAL_TIME/20):1.2f} blocks/s")
        print(f"Rate of climb: {(self.optimized_position[1] - self.starting_height) / (self.TOTAL_TIME/20):1.2f} blocks/s")

    
    def plot(self, save_name=None, save_only = False):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        ax1.plot(self.pitches(self.initial_parameters), color='#A5185F', linewidth=1.2, linestyle='--', label='baseline')

        if self.optimized_parameters is not None:
            ax1.plot(self.pitches(self.optimized_parameters), color='#185FA5', linewidth=1.2, label='optimized')

        ax1.set_ylabel("pitch (degrees)")
        ax1.set_xlabel("step within period")
        ax1.legend()

        ax2.plot(*simulate_pitch_list(self.pitches(self.initial_parameters), start_height=self.starting_height), color='#6E0F56', linewidth=1.2, linestyle='--', label='baseline')
        if self.optimized_parameters is not None:
            X, Y = simulate_pitch_list(self.pitches(self.optimized_parameters), start_height=self.starting_height)
            ax2.plot(X, Y, color='#0F6E56', linewidth=1.2, label='optimized')
        ax2.set_ylabel("height")
        ax2.set_xlabel("horizontal distance")
        ax2.legend()
        plt.tight_layout()

        if save_name:
            plt.savefig(f"Solutions/Plots/{save_name}_{self.goal}_{self.TOTAL_TIME}.png")

        if not save_only:  
            plt.show()


    def save_results(self, name):
        
        with open(f"Solutions/Instances/{name}_{self.goal}_{self.TOTAL_TIME}", 'wb') as f:
            dill.dump(self, f)
        
        csv_path = "Solutions/results_leaderboard.csv"

        with open(csv_path, 'r', newline='') as f:  # Read existing entries to check for duplicates
            reader = csv.reader(f)
            next(reader)  # skip header
            existing = {(row[0], row[1], row[2]) for row in reader}

        key = (name, str(self.TOTAL_TIME), str(self.goal))
        if key in existing:
            with open(csv_path, 'r', newline='') as f:
                rows = list(csv.reader(f))
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in rows:
                    if (row[0], row[1], row[2]) != key:
                        writer.writerow(row)
        
        with open(csv_path, 'a', newline='') as c:
            writer = csv.writer(c)
            writer.writerow([
                name,
                self.TOTAL_TIME,
                self.goal,

                np.round(self.optimized_position[0], 1),
                np.round(self.optimized_position[1] - self.starting_height, 1),
                np.round(self.optimized_position[0] / (self.TOTAL_TIME / 20), 2),
                np.round((self.optimized_position[1] - self.starting_height) / (self.TOTAL_TIME / 20), 2),
            ])


    @classmethod
    def load_results(cls, filename):
        with open(f"Solutions/Instances/{filename}", 'rb') as f:
            return dill.load(f)





    
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
    


logic_dict = {
    "Rectangles": [Rectangles_pitch_function,
                   [[0, 300],   # descent_time
                    [0, 90],    # descent_angle
                    [0, 300],  # ascent_time
                    [-90, 0]     # up_angle
                    ],
                    [160, 33, 120, -33]],

    "RSBoi": [RSBoi_pitch_function,
              [[0, 300],  # descent_time
               [0, 90],   # descent_angle
               [-90, 0],  # up_angle
               [0, 10]    # pitch change rate
               ],
              [120, 33, -50, 0.5]],

    "Inverse_RSBoi": [Inverse_RSBoi_pitch_function,
                    [[0, 90],   # down_angle
                    [-10, 0],  # pitch_change_rate
                    [0, 300],  # ascent_time
                    [-90, 0]   # ascent_angle
                    ],
                    [50, -0.5, 120, -33]],
    }



def OptimizeLogic(logic, N=10, total_time=2000, starting_height=150, verbose=False):
        
    print(f"\nOptimizing for logic: {logic}")
    solution = SolutionTemplate(logic_dict[logic][0], parameters_range=logic_dict[logic][1], initial_parameters=logic_dict[logic][2])
    solution.set_total_time(total_time)
    solution.set_starting_height(starting_height)

    solution.set_goal('max_height')
    solution.best_optimized(N=N)
    solution.set_verbose(verbose)
    solution.print_results()
    solution.plot(save_name=logic, save_only=True)
    solution.save_results(logic)

    solution.set_goal('max_distance')
    solution.best_optimized(N=N)
    solution.set_verbose(verbose)
    solution.print_results()
    solution.plot(save_name=logic, save_only=True)
    solution.save_results(logic)


if __name__ == "__main__":

    OptimizeLogic("Inverse_RSBoi", N=10)
