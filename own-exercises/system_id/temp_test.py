import argparse
import time
import tclab

def main(args):
    TCLab = tclab.setup(connected=False, speedup=10)
    with TCLab() as lab:
        for t in tclab.clock(30):
            lab.Q1(100)
            print(f"Time: {t}s, Temp 1: {lab.T1} °C, Temp 2: {lab.T2} °C. Labtime: {tclab.labtime.time()}s")

    if args.virtual:
        speedup_factor = 12.5
        lab = tclab.setup(connected=False, speedup=speedup_factor)
    else:
        speedup_factor = 1
        lab = tclab.setup(connected=True)

    lab = lab()

    tclab.labtime.reset()
    for t in range(30):
        lab.Q1(100)
        print(f"Time: {t}s, Temp 1: {lab.T1} °C, Temp 2: {lab.T2} °C. Labtime: {tclab.labtime.time()}s")
        time.sleep(1 / speedup_factor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--virtual', action='store_true', help='Use virtual TCLab')
    args = parser.parse_args()
    main(args)