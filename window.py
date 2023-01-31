#	project-e find the most efficient keyboard layout using the genetic algorithm
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

# standart libraries
import copy as cp

# 3rd party libraries
import numpy as np

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QThread, pyqtSlot

# local modules
import gui
from main import MainWorker

# inherited from auto generated qt-designer class to be used as an object in setup_mainwindow
class UIObject(gui.Ui_MainWindow):

	def __init__(self, MainWindow):
		super().setupUi(MainWindow)

class SetupMainwindow(QMainWindow):

	def __init__(self):
		super().__init__()

		self.ui = UIObject(self)

		self.runcount = 0
		self.ui.stopButton.setDisabled(True)
		self.ui.saveButton.setDisabled(True)

		self.ui.startButton.clicked.connect(self.startButtonClicked)
		self.ui.stopButton.clicked.connect(self.stopButtonClicked)
		self.ui.saveButton.clicked.connect(self.saveButtonClicked)

	# signal: sendSaved
	@pyqtSlot(float, np.ndarray)
	def receiveSaved(self, score, layout):
		self.savedscore = cp.copy(score)
		self.savedlayout = layout.copy()

	# signal: updateGenCurrent
	@pyqtSlot(float)
	def workerUpdateGenCurrent(self, val):
		self.ui.genCurrentLabel.setText("current: {:.0f}".format(val))

	# signal: updateGenCount
	@pyqtSlot(int)
	def workerUpdateGenCount(self, val):
		self.gencount = val
		self.ui.genCountLabel.setText(f"gen: {self.gencount}")

	# signal: updateKeys
	@pyqtSlot(np.ndarray)
	def workerUpdateKeys(self, keys):
		for i,v in enumerate(keys):
			self.ui.__dict__[f"key{i}"].setText(v)

	# signal: updateGenLast
	@pyqtSlot(float)
	def workerUpdateGenLast(self, val):
		self.ui.genLastLabel.setText("last: {:.0f}".format(val))

	# signal: updateGenMin
	@pyqtSlot(float)
	def workerUpdateGenMin(self, val):
		self.ui.genMinLabel.setText("min: {:.0f}".format(val))

	# signal: updateGenMax
	@pyqtSlot(float)
	def workerUpdateGenMax(self, val):
		self.ui.genMaxLabel.setText("max: {:.0f}".format(val))

	# signal: updatePlot
	@pyqtSlot(int, float)
	def workerUpdatePlot(self, i, val):
		self.genPlotItem = self.ui.genPlot.getPlotItem()
		self.genPlotItem.setDownsampling(mode='peak')
		self.genPlotItem.setClipToView(True)
		plot = self.genPlotItem.plot()

		if i == 0:
			self.gendata = np.empty(100)

		self.gendata[i] = val
		if i+1 >= self.gendata.shape[0]:
			tmp = self.gendata
			self.gendata = np.empty(self.gendata.shape[0] * 2)
			self.gendata[:tmp.shape[0]] = tmp

		plot.setData(self.gendata[:i+1])

	# signal: updateProgressBar
	@pyqtSlot(int)
	def workerUpdateProgress(self, val):
		self.ui.progressBar.setValue(val)

	# signal: started
	@pyqtSlot()
	def workerStarted(self):
		print("worker started")

		if self.runcount > 0:
			self.genPlotItem.clear()

		self.runcount += 1
		self.ui.startButton.setDisabled(True)
		self.ui.stopButton.setEnabled(True)
		self.ui.saveButton.setDisabled(True)

	#signal: finished
	@pyqtSlot()
	def workerFinished(self):
		print("worker finished")

		self.ui.genCountLabel.setText(f"gen: {self.gencount}")
		self.ui.progressBar.setValue(0)

		self.ui.startButton.setEnabled(True)
		self.ui.stopButton.setDisabled(True)
		self.ui.saveButton.setEnabled(True)

		self.receiveSaved(self.worker.gen_score_min, self.worker.gen_score_min_layout)

		self.thread.quit()
		self.thread.wait()

	def startButtonClicked(self):

		if self.ui.genValueBox.text() == "":
			return

		# create the worker and a qthread instance and move the worker to the said instance
		self.thread = QThread()
		self.worker = MainWorker(int(self.ui.genValueBox.text()))
		self.worker.moveToThread(self.thread)

		self.worker.started.connect(self.workerStarted)
		self.worker.finished.connect(self.workerFinished)

		# connect gui signals
		self.worker.updateGenCount.connect(self.workerUpdateGenCount)
		self.worker.updateGenCurrent.connect(self.workerUpdateGenCurrent)
		self.worker.updateKeys.connect(self.workerUpdateKeys)
		self.worker.updateGenLast.connect(self.workerUpdateGenLast)
		self.worker.updateGenMin.connect(self.workerUpdateGenMin)
		self.worker.updateGenMax.connect(self.workerUpdateGenMax)
		self.worker.updatePlot.connect(self.workerUpdatePlot)
		self.worker.updateProgressBar.connect(self.workerUpdateProgress)

		self.worker.sendSaved.connect(self.receiveSaved)
		self.thread.started.connect(self.worker.work)
		self.thread.start()

	def stopButtonClicked(self):
		self.ui.genCountLabel.setText("stopping...")
		self.worker.stop()

	def saveButtonClicked(self):

		print("saving...")
		with open("layout.txt", "a", encoding='utf-8') as f:

			f.write(f"\nscore:{round(self.savedscore)}")
			f.write(f"\nlayout:\n{self.savedlayout.reshape(3,12)}\n")
		print("saved!")

