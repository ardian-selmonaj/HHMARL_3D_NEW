import gymnasium as gym
import subprocess
import time
import os
from typing import NamedTuple, Tuple

LOADED_MESSAGE = "loading cities done"
LOADED_MESSAGE1 = "Starting hard-coded terrain presampling"

class FlightGearVisualizer(object):
    """
    Class for visualising aircraft using the FlightGear simulator.

    This visualiser launches FlightGear and (by default) waits for it to
    launch. A Figure is also displayed (by creating its own FigureVisualiser)
    which is used to display the agent's actions.
    """

    TYPE = "socket"
    DIRECTION = "in"
    RATE = 60
    SERVER = ""
    PORT = 5550
    PROTOCOL = "udp"
    LOADED_MESSAGE = "loading cities done"
    LOADED_MESSAGE1 = "Starting hard-coded terrain presampling"
    FLIGHTGEAR_TIME_FACTOR = 1  # sim speed relative to realtime, higher is faster
    TIME = "morning"

    def __init__(
        self, sim, print_props, block_until_loaded=True
    ):
        """
        Launches FlightGear in subprocess and starts figure for plotting actions.

        :param sim: Simulation that will be visualised
        :param aircraft: Aircraft to be loaded in FlightGear for visualisation
        :param print_props: collection of Propertys to be printed to Figure
        :param block_until_loaded: visualiser will block until it detects that
            FlightGear has loaded if True.
        """
        self.configure_simulation_output(sim)
        self.print_props = print_props
        self.flightgear_process = self._launch_flightgear(sim.get_aircraft())
        if block_until_loaded:
            self._block_until_flightgear_loaded()

    def plot(self) -> None:
        """
        Updates a 3D plot of agent actions.
        """
        pass

    @staticmethod
    def _launch_flightgear(aircraft):
        cmd_line_args = FlightGearVisualiser._create_cmd_line_args(
            aircraft.flightgear_id
        )
        gym.logger.info(f'Subprocess: "{cmd_line_args}"')
        flightgear_process = subprocess.Popen(
            cmd_line_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        gym.logger.info("Started FlightGear")
        return flightgear_process

    def configure_simulation_output(self, sim):
        sim.enable_flightgear_output()
        sim.set_simulation_time_factor(self.FLIGHTGEAR_TIME_FACTOR)

    @staticmethod
    def _create_cmd_line_args(aircraft_id: str):
        # FlightGear doesn't have a 172X model, use the P instead
        if aircraft_id == "c172x":
            aircraft_id = "c172p"

        flightgear_cmd = "fgfs"
        aircraft_arg = f"--aircraft={aircraft_id}"
        flight_model_arg = (
            "--native-fdm=" + f"{FlightGearVisualiser.TYPE},"
            f"{FlightGearVisualiser.DIRECTION},"
            f"{FlightGearVisualiser.RATE},"
            f"{FlightGearVisualiser.SERVER},"
            f"{FlightGearVisualiser.PORT},"
            f"{FlightGearVisualiser.PROTOCOL}"
        )
        flight_model_type_arg = "--fdm=" + "external"
        disable_ai_arg = "--disable-ai-traffic"
        disable_live_weather_arg = "--disable-real-weather-fetch"
        time_of_day_arg = "--timeofday=" + FlightGearVisualiser.TIME
        return (
            flightgear_cmd,
            aircraft_arg,
            flight_model_arg,
            flight_model_type_arg,
            disable_ai_arg,
            disable_live_weather_arg,
            time_of_day_arg,
        )

    def _block_until_flightgear_loaded(self):
        while True:
            msg_out = self.flightgear_process.stdout.readline().decode()
            if self.LOADED_MESSAGE in msg_out or self.LOADED_MESSAGE1 in msg_out:
                time.sleep(5)
                print("FlightGear Loading Complete")
                break
            else:
                time.sleep(0.1)

    def close(self):
        if self.flightgear_process:
            self.flightgear_process.kill()


class FlightGearBasic(object):
    def __init__(self, env, agent_sim) -> None:
        
        self.env = env
        self.sim = agent_sim

        #self.load_fg()
    
    def load_fg(self):
        cmd = "fgfs"
        aircraft = "--aircraft=c172p"
        cmd_line_args = (cmd, aircraft)
        flightgear_process = subprocess.Popen(
            cmd_line_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._block_until_flightgear_loaded(flightgear_process)
        

    def _block_until_flightgear_loaded(self, flightgear_process):
        while True:
            msg_out = flightgear_process.stdout.readline().decode()
            if LOADED_MESSAGE in msg_out or LOADED_MESSAGE1 in msg_out:
                time.sleep(5)
                print("FlightGear Loading Complete")
                break
            else:
                time.sleep(0.1)