import gui

import numpy as np

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import *

class setup_mainwindow(gui.Ui_MainWindow):

	def __init__(self, MainWindow, MainFunc):
		super().__init__()
		self.program_state = False
		self.runcount = 0
		self.MainFunc = MainFunc
		self.MainWindow = MainWindow
		self.setupUi(MainWindow)

	def setupUi(self, MainWindow):
		super().setupUi(MainWindow)
		self.prog_stop_btn.setEnabled(False)

		self.prog_start_btn.clicked.connect(self.event_prog_start_btn_clicked)
		self.prog_stop_btn.clicked.connect(self.event_prog_stop_btn_clicked)
		self.prog_save_btn.clicked.connect(self.event_prog_save_btn_clicked)

	def event_thread_update_currgen(self, val):
		self.gen_label.setText("current: {:.0f}".format(val))

	def event_thread_update_gencount(self, val):
		self.gencount = val
		self.gencount_label.setText(f"gen: {self.gencount}")

	def event_thread_update_keys(self, keys):
		for i,v in enumerate(keys):
			self.__dict__[f"key{i}"].setText(v)
		
	def event_thread_update_lastgen(self, val):
		self.lastgen_label.setText("last: {:.0f}".format(val))

	def event_thread_update_mingen(self, val):
		self.mingen_label.setText("min: {:.0f}".format(val))

	def event_thread_update_maxgen(self, val):
		self.maxgen_label.setText("max: {:.0f}".format(val))

	def event_thread_update_plot(self, i, val):
		self.gvplot_item = self.genVisualization.getPlotItem()
		self.gvplot_item.setDownsampling(mode='peak')
		self.gvplot_item.setClipToView(True)
		self.gvplot = self.gvplot_item.plot()
		
		if i == 0:
			self.gendata = np.empty(100)

		self.gendata[i] = val
		if i+1 >= self.gendata.shape[0]:
			tmp = self.gendata
			self.gendata = np.empty(self.gendata.shape[0] * 2)
			self.gendata[:tmp.shape[0]] = tmp
		
		self.gvplot.setData(self.gendata[:i+1])

	def event_thread_update_progress(self, val):
		self.progressBar.setValue(val)

	def event_prog_stop_btn_clicked(self):
		self.thread.quit_flag = True
		self.gencount_label.setText("stopping...")

	def event_thread_finished(self):
		print("event finished")
		self.program_state = False

		self.gencount_label.setText(f"gen: {self.gencount}")
		self.progressBar.setValue(0)	
		self.prog_start_btn.setEnabled(True)
		self.prog_stop_btn.setEnabled(False)
		self.prog_save_btn.setEnabled(True)


	def event_thread_started(self):
		print("event started")
		self.program_state = True

		if self.runcount > 0:	
			self.gvplot_item.clear()
		
		self.runcount += 1
		self.prog_start_btn.setDisabled(True)
		self.prog_stop_btn.setEnabled(True)
		self.prog_save_btn.setDisabled(True)

	def event_prog_start_btn_clicked(self):
		
		if self.program_state == True:
			return
	
		if self.gen_value_box.text() == "":
			return
		
		# create a qthread instance and start it
		self.thread = sub_qthread(self.MainFunc, int(self.gen_value_box.text()), parent=self.MainWindow)

		self.thread.finished.connect(self.event_thread_finished)
		self.thread.started.connect(self.event_thread_started)

		self.thread.start()
		
		# connect gui signals
		self.thread.update_gencount.connect(self.event_thread_update_gencount)
		self.thread.update_currgen.connect(self.event_thread_update_currgen)
		self.thread.update_keys.connect(self.event_thread_update_keys)
		self.thread.update_lastgen.connect(self.event_thread_update_lastgen)
		self.thread.update_mingen.connect(self.event_thread_update_mingen)
		self.thread.update_maxgen.connect(self.event_thread_update_maxgen)
		self.thread.update_plot.connect(self.event_thread_update_plot)
		self.thread.update_progress.connect(self.event_thread_update_progress)

	def event_prog_save_btn_clicked(self):
		pass

class sub_qthread(QThread):
	update_gencount = pyqtSignal(int)
	update_currgen = pyqtSignal(float)
	update_keys = pyqtSignal(np.ndarray)
	update_lastgen = pyqtSignal(float)
	update_mingen = pyqtSignal(float)
	update_maxgen = pyqtSignal(float)
	update_plot = pyqtSignal(int, float)
	update_progress = pyqtSignal(int)

	def __init__(self, MainFunc, GenValue, parent=None):
		super().__init__(parent)
		self.quit_flag = False
		self.MainFunc = MainFunc
		self.GenValue = GenValue

	def run(self):
		self.MainFunc(self, self.GenValue)
	
