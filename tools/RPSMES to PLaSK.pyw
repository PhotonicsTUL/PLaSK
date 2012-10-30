#!/usr/bin/python
import dan2xpl

import sys
import os
import traceback

from Tkinter import *
import tkFileDialog

class App(Frame):
    def run(self):
        if self.iname is None:
            return
    
        sys.stdout = self
        sys.stderr = self
        
        dest_dir = os.path.dirname(self.iname)
        try:
            read = dan2xpl.read_dan(self.iname)
            name = os.path.join(dest_dir,read[0])
            dan2xpl.write_xpl(name, *read[1:])
        except Exception:
            traceback.print_exc()
        else:
            print "Done!"
        
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def open_file(self):
        self.iname = tkFileDialog.askopenfilename(defaultextension=".dan", filetypes=[("RPSMES", "*.dan"), ("All files", "*")])
        self.run()
        
    def add_buttons(self):
        self.button = Button(self)
        self.button["text"] = "Open"
        self.button["command"] = self.open_file
        self.button.pack(side=TOP)

    def write(self, txt):
        self.text1.insert(INSERT, txt)

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack(fill=BOTH, expand=1)
        frame = Frame(self)
        frame.pack(side=TOP, fill=BOTH, expand=1)
        scrollbar = Scrollbar(frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.text1 = Text(frame, yscrollcommand=scrollbar.set)
        self.text1.pack(side=LEFT, fill=BOTH, expand=1)
        scrollbar.config(command=self.text1.yview)
        try:
            self.iname = sys.argv[1]
        except IndexError:
            self.iname = None
            self.add_buttons()
        else:
            self.run()

root = Tk()
app = App(master=root)
app.mainloop()
