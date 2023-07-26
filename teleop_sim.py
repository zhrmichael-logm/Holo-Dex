import hydra
import time
from processes import get_camera_stream_processes, get_detector_processes, get_teleop_process, get_simulation_process

@hydra.main(version_base = '1.2', config_path='configs', config_name='teleop')
def main(configs):    

    simulation_process = get_simulation_process()
    teleop_process = get_teleop_process(configs)

    # Starting all the processes
    simulation_process.start()
    time.sleep(3)

    # Teleop process
    teleop_process.start()

    simulation_process.join()
    teleop_process.join()

if __name__ == '__main__':
    main()