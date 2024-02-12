import os

import plask

plask.print_log('important', 'MODULE')
print(f"IMPORTANT     : {__file__}:6 : MODULE")
print(f"IMPORTANT     : {os.path.basename(__file__)}:7 : MODULE")

#
