from pyboard import Pyboard

class Act():
    def __init__(self):
        self.baudrate = 115200 # This is the standard board rate to communicate with the Lego Hub
        self.device = '/dev/ttyACM0' 
        self.wait= 0
        self.pyb = Pyboard(self.device, self.baudrate, self.wait)
        self.pyb.enter_raw_repl()
        return
    
    def add_to_parser(self, parser):
        parser.add_argument('--actiondir', help='Specify act/movement control file', default=None)
        
    async def run(self):
        try:
            while True:
                say_text_wait(pyb, 'Starting')
                await asyncio.sleep(1)
        
        # Only terminate when program is to be halted.
        finally:
            self.pyb.exit_raw_repl()
            self.pyb.close()
        return
    
    def say_text_wait(pyb, sentence):
        command = f"""\
from hub import light_matrix
import runloop
async def main():
    await light_matrix.write('{sentence}')

runloop.run(main())
"""
        pyb.exec(command)