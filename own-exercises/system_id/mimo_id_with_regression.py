import csv
import numpy as np
import scipy.optimize as opt

import matplotlib.pyplot as plt

# Load data
data = "own-exercises/system_id/recorded_data.csv"

# 6 unknown parameters
# R_cond_1, R_rad_1, R_cond_2, R_rad_2, R_cond_1_2, R_rad_1_2

class ModelParameters():
    def __init__(self, np_array):
        self.R_cond_1 = np_array[0]
        self.R_rad_1 = np_array[1]
        self.R_cond_2 = np_array[2]
        self.R_rad_2 = np_array[3]
        self.R_cond_1_2 = np_array[4]
        self.R_rad_1_2 = np_array[5]

    def to_np_array(self):
        return np.array([self.R_cond_1, self.R_rad_1, self.R_cond_2, self.R_rad_2, self.R_cond_1_2, self.R_rad_1_2])

    

alpha_1 = 0.01 # W / % heater
alpha_2 = 0.005 # W / % heater

m = 0.004 # kg
c_p = 500 # J / (kg K)

T_inf_degC = 23 # °C

def predict_temperatures(temp_initial, time_values, control_values, parameters: ModelParameters):
    predicted_temperatures = [[temp_initial[0], temp_initial[1]]]
    for i in range(1, len(time_values)):
        dt = time_values[i] - time_values[i - 1]
        T1 = predicted_temperatures[-1][0]
        T2 = predicted_temperatures[-1][1]

        # I = V / R. Therefore W = T / R
        W_cond_1 = (T1 - T_inf_degC) / parameters.R_cond_1
        W_rad_1 = (T1**4 - T_inf_degC**4) / parameters.R_rad_1
        W_cond_2 = (T2 - T_inf_degC) / parameters.R_cond_2
        W_rad_2 = (T2**4 - T_inf_degC**4) / parameters.R_rad_2

        W_cond_1_2 = (T1 - T2) / parameters.R_cond_1_2
        W_rad_1_2 = (T1**4 - T2**4) / parameters.R_rad_1_2

        T1_dot = (alpha_1 * control_values[i][0] - W_cond_1 - W_rad_1 - W_cond_1_2 - W_rad_1_2) / (m * c_p)
        T2_dot = (alpha_2 * control_values[i][1] - W_cond_2 - W_rad_2 + W_cond_1_2 + W_rad_1_2) / (m * c_p)

        predicted_temperatures.append([T1 + T1_dot * dt, T2 + T2_dot * dt])
    return predicted_temperatures

def callback(params):
    print("Current params:", params)

def cost_function(parameters, time_values, temperature_values, control_values):
    parameters = ModelParameters(parameters)
    predicted_temperatures = predict_temperatures(temperature_values[0], time_values, control_values, parameters)
    error = 0
    for i in range(len(time_values)):
        error += (temperature_values[i][0] - predicted_temperatures[i][0])**2 + (temperature_values[i][1] - predicted_temperatures[i][1])**2
    return error / len(time_values)

def main():
    time_values = []
    temperature_values = []
    control_values = []

    with open(data, mode='r') as recorded_data_file:
        recorded_data_reader = csv.reader(recorded_data_file)
        next(recorded_data_reader)
        for row in recorded_data_reader:
            time_values.append(float(row[0]))
            temperature_values.append([float(row[1]), float(row[2])])
            control_values.append([float(row[3]), float(row[4])])

    sigma = 5.67e-8 # W / (m^2 K^4)
    sample_area = 1e-3 # m^2
    sample_conductivity = 5 # W / (m^2 K)

    initial_guess_cond = 1 / (sample_conductivity * sample_area)
    initial_guess_rad = 1 / (sigma * sample_area)

    initial_guess = np.array([initial_guess_cond, initial_guess_rad, initial_guess_cond, initial_guess_rad, initial_guess_cond, initial_guess_rad])
    options = {
        "maxiter": 2000,
    }
    result = opt.minimize(cost_function, initial_guess, args=(time_values, temperature_values, control_values), options=options, callback=callback)
    print(result)

    print(result.x)

    final_parameters = ModelParameters(result.x)

    predicted_temperatures = predict_temperatures(temperature_values[0], time_values, control_values, final_parameters)

    fig, axs = plt.subplots(2)
    # Plot predicted and recorded temperatures with time on the x-axis
    axs[0].plot(time_values, [q[0] for q in predicted_temperatures], label="Predicted Temperature for Heater 1")
    axs[0].plot(time_values, [q[1] for q in predicted_temperatures], label="Predicted Temperature for Heater 2")

    axs[0].plot(time_values, [q[0] for q in temperature_values], label="Recorded Temperature for Heater 1")
    axs[0].plot(time_values, [q[1] for q in temperature_values], label="Recorded Temperature for Heater 2")
    axs[0].legend()

    # Plot control values with time on the x-axis
    axs[1].plot(time_values, [c[0] for c in control_values], label="Control Value for Heater 1")
    axs[1].plot(time_values, [c[1] for c in control_values], label="Control Value for Heater 2")
    axs[1].legend()

    plt.show()



if __name__ == "__main__":
    main()