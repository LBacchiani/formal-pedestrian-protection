import threading
import time


class SyncSimulation(object):
    def __init__(self, world, tick_delay = 0.05):
        self.run = True
        self.world = world
        self.tick_delay = tick_delay

    def set_synchronous_mode(self, synchronous_mode_flag):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode_flag
        if synchronous_mode_flag:
            settings.fixed_delta_seconds = self.tick_delay
        else: 
            settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.world.tick()


    def simulation_ticker(self):
        self.set_synchronous_mode(True)
        while self.run:
            time.sleep(self.tick_delay)
            self.world.tick()
        self.set_synchronous_mode(False)

    def stop_simulation(self):
        self.run = False

    def run_sync_simulation(self):
        threading.Thread(target=self.simulation_ticker).start()
