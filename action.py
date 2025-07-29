from pyboard import Pyboard
import time
import kbSingleton

class Act():
    def __init__(self):
        print("Action init func")
        self.kb = kbSingleton.kb_instance
        self.baudrate = 115200 # This is the standard board rate to communicate with the Lego Hub
        self.device = '/dev/ttyACM0' 
        self.wait= 0
        self.pyb = Pyboard(self.device, self.baudrate, self.wait)
        self.pyb.enter_raw_repl()
        return
    
    def add_to_parser(self, parser):
        parser.add_argument('--actiondir', help='Specify act/movement control file', default=None)
        
    def run(self):
        print("action run func")
        try:
            # Need to establish pair once, hence outside of loop
            self.init_pair(self.pyb)
            while True:
                #move towards the line
                line = self.kb.get('closest_line')
                screen_center = self.kb.get('screen_center')
                if (line is not None) & (screen_center is not None):
                    line_midpoint = (line[0]+(line[1]-line[0])/2), (line[2]+(line[3]-line[2])/2)
                    self.move_pair_to_point(self.pyb, line_midpoint, screen_center)
                    time.sleep(0.5)
        
        # Only terminate when program is to be halted.
        finally:
            self.pyb.exit_raw_repl()
            self.pyb.close()
        return
    
    def say_text_wait(self, pyb, sentence):
        command = f"""\
from hub import light_matrix
import runloop
async def main():
    await light_matrix.write('{sentence}')

runloop.run(main())
"""
        pyb.exec(command)
        
    def init_pair(self, pyb):
        command = f"""\
from hub import port
import motor_pair, time

motor_pair.pair(motor_pair.PAIR_1, port.A, port.E)
"""
        pyb.exec(command)
    
    def move_pair_to_point(self, pyb, point, center):
        # Control logic to adjust direction to face point based on current middle of screen.
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        print("dx = ", dx, "dy: ", dy, "point: ", point, "center: ", center)
        normalised_dx = dx / center[0]
        print("normalised dx: ", normalised_dx)
        steering = int(normalised_dx * 100)
        print(steering)
        # speed 
        speed = 1000
        
        command = f"""motor_pair.move_for_time(motor_pair.PAIR_1, {speed}, 0 , velocity= 280)"""
        #pyb.exec(command.encode())
