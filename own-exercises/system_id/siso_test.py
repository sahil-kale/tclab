import argparse
import time
import tclab
import matplotlib.pyplot as plt

ID_ONTIME_START = 20
ID_ONTIME_END = 300
TOTAL_TIME = 500
ID_PERCENT = 100

T_inf = 22 # °C

def get_control_input(t) -> list[float]:
    control = [0, 0]
    if (t >= ID_ONTIME_START) and (t < ID_ONTIME_END):
        control = [ID_PERCENT, 0]
    
    return control

def run_model(dt, u, Temperature: list[float], ) -> list[float]:
    Q1 = u[0]
    Q2 = u[1] # not used for now...

    # Parameters
    alpha = 0.01 # W / % heater
    U = 10 # W / (m^2 K)
    A = 1.2e-3 # m^2
    m = 0.004 # kg
    c_p = 500 # J / (kg K)
    emissivity = 0.9
    sigma = 5.67e-8 # W / (m^2 K^4)

    T1 = Temperature[0]
    # round T1 to 2 decimal places
    T1 = round(T1, 2)
    T2 = Temperature[1]

    T1_dot = (alpha * Q1 - U * A * (T1 - T_inf) - emissivity * sigma * A * ((T1 + 273)**4 - (T_inf + 273)**4) ) / (m * c_p)

    return [T1 + T1_dot * dt, T2] # basic integral

def main(args):
    TCLab = None
    if args.virtual:
        TCLab = tclab.setup(connected=False, speedup=15)
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

    predicted_temperature_values = [[T_inf, T_inf]]
    for i in range(len(time_vectors)):
        dt = time_vectors[i] - time_vectors[i - 1] if i > 0 else 0
        u = control_values[i]
        if i > 0:
            predicted_temperature_values.append(run_model(dt, u, predicted_temperature_values[-1]))
        print(f"Time: {time_vectors[i]}s, Temp 1: {predicted_temperature_values[i][0]} °C, Temp 2: {predicted_temperature_values[i][1]} °C")

    # Plot s

    plt.plot(time_vectors, [x[0] for x in recorded_temperature_values], label='Recorded Temp 1')
    plt.plot(time_vectors, [x[0] for x in predicted_temperature_values], label='Predicted Temp 1')
    plt.legend()
    plt.show()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--virtual', action='store_true', help='Use virtual TCLab')
    args = parser.parse_args()
    main(args)
        
