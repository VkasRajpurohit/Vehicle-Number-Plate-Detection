import os
import logging as lg

# Create and configure logger
lg.basicConfig(filename=os.path.join(r"Logs\\logfile.log"),
               filemode='w',
               level=lg.INFO,
               format='%(asctime)s %(levelname)s: %(message)s',
               datefmt='%Y-%m-%d %H:%M:%S')
