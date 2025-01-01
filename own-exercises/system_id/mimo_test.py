import argparse
import time
import tclab
import matplotlib.pyplot as plt
import numpy as np


TOTAL_TIME = 600

T_inf_degC = 23 # °C

def get_control_input(t) -> list[float]:
    q1_control = np.zeros(TOTAL_TIME+1)
    q2_control = np.zeros(TOTAL_TIME+1)
    
    q1_control[10:200] = 80
    q1_control[200:280] = 20
    q1_control[280:400] = 70
    q1_control[400:] = 50

    q2_control[120:320] = 100
    q2_control[320:520] = 10
    q2_control[520:] = 80

    i = int(t)

    return [q1_control[i], q2_control[i]]

def run_model(dt, u, Temperature: list[float], ) -> list[float]:
    Q1 = u[0]
    Q2 = u[1]

    # Parameters
    alpha_1 = 0.01 # W / % heater
    alpha_2 = 0.005 # W / % heater
    U = 10 # W / (m^2 K)
    A = 1e-3 # m^2
    A_s = 2e-4 # m^2
    m = 0.004 # kg
    c_p = 500 # J / (kg K)
    emissivity = 0.9
    sigma = 5.67e-8 # W / (m^2 K^4)

    T1 = Temperature[0]
    T2 = Temperature[1]

    W_cond_1 = U * A * (T1 - T_inf_degC)
    W_rad_1 = emissivity * sigma * A * ((T1 + 273)**4 - (T_inf_degC + 273)**4)
    W_cond_2 = U * A * (T2 - T_inf_degC)
    W_rad_2 = emissivity * sigma * A * ((T2 + 273)**4 - (T_inf_degC + 273)**4)

    W_cond_1_2 = U * A_s * (T1 - T2)
    W_rad_1_2 = emissivity * sigma * A_s * ((T1 + 273)**4 - (T2 + 273)**4)

    T1_dot = (alpha_1 * Q1 - W_cond_1 - W_rad_1 - W_cond_1_2 - W_rad_1_2) / (m * c_p)
    T2_dot = (alpha_2 * Q2 - W_cond_2 - W_rad_2 + W_cond_1_2 + W_rad_1_2) / (m * c_p)

    return [T1 + T1_dot * dt, T2 + T2_dot * dt] 
    

def main(args):
    TCLab = None
    if args.virtual:
        TCLab = tclab.setup(connected=False, speedup=500)
    else:
        TCLab = tclab.setup(connected=True)

    time_vectors = []
    control_values = []
    recorded_temperature_values = []

    with TCLab() as lab:
        for t in tclab.clock(TOTAL_TIME, adaptive=False):
            control = get_control_input(t)
            lab.Q1(control[0])
            lab.Q2(control[1])
            control_values.append(control)
            recorded_temperature_values.append([lab.T1, lab.T2])
            time_vectors.append(t)
            print(f"Time: {t}s, Temp 1: {lab.T1} °C, Temp 2: {lab.T2} °C. Labtime: {tclab.labtime.time()}s")

    # write the recorded data to a file
    import csv
    file_name = f'recorded_data{"_virtual" if args.virtual else ""}.csv'
    with open('recorded_data.csv', mode='w') as recorded_data_file:
        recorded_data_writer = csv.writer(recorded_data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        recorded_data_writer.writerow(['Time', 'Temp 1', 'Temp 2', 'Control 1', 'Control 2'])
        for i in range(len(time_vectors)):
            recorded_data_writer.writerow([time_vectors[i], recorded_temperature_values[i][0], recorded_temperature_values[i][1], control_values[i][0], control_values[i][1]])

    predicted_temperature_values = [[T_inf_degC, T_inf_degC]]
    for i in range(len(time_vectors)):
        dt = time_vectors[i] - time_vectors[i - 1] if i > 0 else 0
        u = control_values[i]
        if i > 0:
            predicted_temperature_values.append(run_model(dt, u, predicted_temperature_values[-1]))
        print(f"Time: {time_vectors[i]}s, Temp 1: {predicted_temperature_values[i][0]} °C, Temp 2: {predicted_temperature_values[i][1]} °C")

    # create 2 subplots, one for temperature and one for control input
    fig, axs = plt.subplots(2)
    fig.suptitle('Temperature and Control Input')
    axs[0].plot(time_vectors, [x[0] for x in recorded_temperature_values], label='Recorded Temp 1')
    axs[0].plot(time_vectors, [x[0] for x in predicted_temperature_values], label='Predicted Temp 1')
    axs[0].plot(time_vectors, [x[1] for x in recorded_temperature_values], label='Recorded Temp 2')
    axs[0].plot(time_vectors, [x[1] for x in predicted_temperature_values], label='Predicted Temp 2')
    axs[0].legend()
    axs[1].plot(time_vectors, [x[0] for x in control_values], label='Control Input 1')
    axs[1].plot(time_vectors, [x[1] for x in control_values], label='Control Input 2')
    axs[1].legend()

    plt.show()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--virtual', action='store_true', help='Use virtual TCLab')
    args = parser.parse_args()
    main(args)
        
