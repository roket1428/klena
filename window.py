#	project-e find most efficient keyboard layout using genetic algorithm
#		Copyright (C) 2023 roket1428 <meorhan@protonmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# local modules
import gui
from main import main_worker

# standart modules
import copy as cp

# 3rd party libraries
import numpy as np

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

# derived from auto generated qt-designer class
class setup_ui(gui.Ui_MainWindow):

	def __init__(self, MainWindow):
		super().setupUi(MainWindow)

class setup_mainwindow(QMainWindow):

	def __init__(self):
		super().__init__()

		self.ui = setup_ui(self)

		self.runcount = 0

		self.ui.prog_stop_btn.setDisabled(True)
		self.ui.prog_save_btn.setDisabled(True)

		self.ui.prog_start_btn.clicked.connect(self.event_prog_start_btn_clicked)
		self.ui.prog_stop_btn.clicked.connect(self.event_prog_stop_btn_clicked)
		self.ui.prog_save_btn.clicked.connect(self.event_prog_save_btn_clicked)

	@pyqtSlot(float, np.ndarray)
	def receive_saved(self, score, layout):
		self.save_score = cp.copy(score)
		self.save_layout = layout.copy()

	@pyqtSlot(float)
	def event_worker_update_currgen(self, val):
		self.ui.gen_label.setText("current: {:.0f}".format(val))

	@pyqtSlot(int)
	def event_worker_update_gencount(self, val):
		self.gencount = val
		self.ui.gencount_label.setText(f"gen: {self.gencount}")

	@pyqtSlot(np.ndarray)
	def event_worker_update_keys(self, keys):
		for i,v in enumerate(keys):
			self.ui.__dict__[f"key{i}"].setText(v)

	@pyqtSlot(float)
	def event_worker_update_lastgen(self, val):
		self.ui.lastgen_label.setText("last: {:.0f}".format(val))

	@pyqtSlot(float)
	def event_worker_update_mingen(self, val):
		self.ui.mingen_label.setText("min: {:.0f}".format(val))

	@pyqtSlot(float)
	def event_worker_update_maxgen(self, val):
		self.ui.maxgen_label.setText("max: {:.0f}".format(val))

	@pyqtSlot(int, float)
	def event_worker_update_plot(self, i, val):
		self.gvplot_item = self.ui.genVisualization.getPlotItem()
		self.gvplot_item.setDownsampling(mode='peak')
		self.gvplot_item.setClipToView(True)
		gvplot = self.gvplot_item.plot()

		if i == 0:
			self.gendata = np.empty(100)

		self.gendata[i] = val
		if i+1 >= self.gendata.shape[0]:
			tmp = self.gendata
			self.gendata = np.empty(self.gendata.shape[0] * 2)
			self.gendata[:tmp.shape[0]] = tmp

		gvplot.setData(self.gendata[:i+1])

	@pyqtSlot(int)
	def event_worker_update_progress(self, val):
		self.ui.progressBar.setValue(val)

	@pyqtSlot()
	def event_worker_started(self):
		print("worker started")

		if self.runcount > 0:
			self.gvplot_item.clear()

		self.runcount += 1
		self.ui.prog_start_btn.setDisabled(True)
		self.ui.prog_stop_btn.setEnabled(True)
		self.ui.prog_save_btn.setDisabled(True)

	@pyqtSlot()
	def event_worker_finished(self):
		print("worker finished")

		# reset gencount and progressbar
		self.ui.gencount_label.setText(f"gen: {self.gencount}")
		self.ui.progressBar.setValue(0)

		# update button clickablity
		self.ui.prog_start_btn.setEnabled(True)
		self.ui.prog_stop_btn.setDisabled(True)
		self.ui.prog_save_btn.setEnabled(True)

		# save layout and fitness score for the save file button
		self.receive_saved(self.worker.mingen_scr, self.worker.mingen_lyt)

		self.thread.quit()
		self.thread.wait()

	def event_prog_start_btn_clicked(self):

		if self.ui.gen_value_box.text() == "":
			return

		# create the worker and a qthread instance & move worker to thread
		self.worker = main_worker(int(self.ui.gen_value_box.text()))
		self.thread = QThread()
		self.worker.moveToThread(self.thread)

		self.worker.started.connect(self.event_worker_started)
		self.worker.finished.connect(self.event_worker_finished)

		#connect gui signals
		self.worker.update_gencount.connect(self.event_worker_update_gencount)
		self.worker.update_currgen.connect(self.event_worker_update_currgen)
		self.worker.update_keys.connect(self.event_worker_update_keys)
		self.worker.update_lastgen.connect(self.event_worker_update_lastgen)
		self.worker.update_mingen.connect(self.event_worker_update_mingen)
		self.worker.update_maxgen.connect(self.event_worker_update_maxgen)
		self.worker.update_plot.connect(self.event_worker_update_plot)
		self.worker.update_progress.connect(self.event_worker_update_progress)

		self.worker.send_saved.connect(self.receive_saved)
		self.thread.started.connect(self.worker.work)
		self.thread.start()

	def event_prog_stop_btn_clicked(self):
		self.ui.gencount_label.setText("stopping...")
		self.worker.stop()

	def event_prog_save_btn_clicked(self):

		print("saving...")
		with open("layout.txt", "a", encoding='utf-8') as f:

			f.write(f"\nscore:{self.save_score}")
			f.write(f"\nlayout:{self.save_layout.reshape(3,12)}\n")
