#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Energy Saving Simulator
# 
# DB : MS SQL 
# Program Language : Python
#
# kgpark@hdc-icontrols.com
# Oct, 2020

import os
import sys
import pymssql
import numpy as np

import openpyxl

## pyQt5 modules
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QDialog, QVBoxLayout,\
							QHBoxLayout, QTabWidget, QComboBox, QLabel, QGridLayout, QMessageBox,\
							QGroupBox, QRadioButton, QBoxLayout, QDateTimeEdit, QCheckBox
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
 
## matplotlib modules
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec

## time modules
from datetime import datetime as dt
from datetime import timedelta

## developing modules
sys.path.append(r'D:\IPTower\\')
sys.path.append(r'D:\IPTower\CycleControl\\')
sys.path.append(r'D:\IPTower\EnthalpyControl\\')
sys.path.append(r'D:\IPTower\NightPurgeControl\\')
sys.path.append(r'D:\IPTower\AHUOptimulControl\\')
import Cycle_Control as CC
import Enthalpy_Control as EC
import Nightpurge_Control as NC
import AHU_Optimul_Control as AOC

class MainWindow(QWidget):
	os.chdir(r'D:\IPTower\\')  # move directory to current file
	#### Define variables to remember continuously ------------------------------------------	
	# Fix current time as global variable
	currentDateTime = dt.now()
	
	AHU_tag = ['AHU6 (지상3층)', 'AHU6 (지상4층)', 'AHU6 (지상5층)', 'AHU6 (지상6층)', 'AHU6 (지상7층)', 'AHU6 (지상8층)', 'AHU6 (지상9층)']	# 'AHU4 (지상2층)', 
	AHU_tag_all = ['AHU-101', 'AHU-102', 'AHU-103', 'AHU-104', 'AHU-105', 'AHU-106', 'AHU-107', 'AHU-108', 'AHU-109', 'AHU-110', 'AHU-111', 'AHU-112']
	AHU_tag_AOC = ['AHU-105', 'AHU-111', 'AHU-112']# ['AHU-111', 'AHU-112']

	AHUidx_CC = 0
	InputParam_CC = []
	Result_CC = []		## AHU Cycle Control .py result	
	for i in range(len(AHU_tag)):
		InputParam_CC.append({'AHU': AHU_tag[i], 'InitialCoolMonth': 6, 'FinalCoolMonth': 9, 'CycleStatus': [0,0,0,0], 'LowerTemp' : 22, 'UpperTemp' : 26, 'InitialTime': 8, 'FinalTime': 18})
		Result_CC.append({'AHUStatus_measured': None, 'RA_measured': None, 'AHUStatus_controlled': None, 'RA_controlled': None, 'AHUStatus_optimul': None, 'RA_optimul': None, 'PredictedDay': None, 'msg': None})
		
	AHUidx_EC = 0
	InputParam_EC = []
	Saving_tab_EC = []
	Result_EC = []  ## Enthalpy Control .py result
	for i in range(len(AHU_tag)):
		InputParam_EC.append({'AHU': AHU_tag[i],'Train_Month': 4, 'Train_Day': 1})
		Result_EC.append({'ahu_damper_BEMS': None, 'ahu_damper_pred': None,'ahu_damper_control': None, 'MA temp_BEMS': None,
						  'y_prediction_ma': None, 'y_prediction_ma_c': None, 'Enthalpy_BEMS': None, 'Enthalpy_predicton': None,
						  'y_prediction_Enthalpy': None, 'y_prediction_c_1_Enthalpy': None, 'label': None, 'PredictedDay': None, 'msg': None})

	AHUidx_NC = 0
	InputParam_NC = []
	NC_able_tab = []
	NC_saving_tab = []
	Result_NC = []
	for i in range(len(AHU_tag)):
		InputParam_NC.append({'AHU': AHU_tag[i], 'delta_RAOA': 3.0,
							  'initial_time': 20, 'InitialCoolMonth': 5, 'FinalCoolMonth': 10,
							  'end_hour': 5, 'end_minute': 0, 'margin_error': 0.3})
		Result_NC.append({'title': None, 'future_num': None, 'x_label': None,
							'RA_asis': None, 'RA_tobe': None, 'RA_fail': None,
							'OA': None, 'RA_proposal': None,
							'damp_on_best': None, 'Damper_fail': None, 'damp_asis': None,
							'damper_ratio': None, 'AHU_on_proposal': None, 'load_saving': None,
							'msg': None})
		
	AHUidx_AOC = 0
	Result_AOC = []		## AHU Optimul Control .py result	
	InputParam_AOC = []
	for i in range(len(AHU_tag)):
		InputParam_AOC.append({'AHU': AHU_tag[i], 'SSRadioBtn': 'Start', 'InitialCoolMonth': 6, 'FinalCoolMonth': 9, 'OpeningTime': 9, 'ClosingTime': 18, 'AHU_TargetTemp' : 24, 'AHU_TempTolerance' : 0.15})		
		Result_AOC.append({'AHU_Start': None, 'AHU_Stop': None, 'PredictedDate': None, 'msg': None})
		if currentDateTime.hour > 14:
			InputParam_AOC[-1]['SSRadioBtn'] = 'Stop' 		
	#if currentDateTime.month < 4 or currentDateTime.month > 10:
	#	InputParam_CAOC['TargetTemp'] = 24
		
	## For using another class 			
	fig_CC = plt.Figure(figsize=(13,9))
	canvas_CC = FigureCanvas(fig_CC)	# figure - canvas link
	
	fig_EC = plt.figure(figsize=(13,9))
	canvas_EC = FigureCanvas(fig_EC)	# figure - canvas link
	
	fig_NC = plt.figure(figsize=(13,9))
	canvas_NC = FigureCanvas(fig_NC)
	
	fig_AOC = plt.figure(figsize=(13,9))
	canvas_AOC = FigureCanvas(fig_AOC)	# figure - canvas link
	
	# btn layout			
	canvasLayout_CC = QVBoxLayout()
	canvasLayout_CC.addWidget(canvas_CC)
	
	canvasLayout_EC = QVBoxLayout()
	canvasLayout_EC.addWidget(canvas_EC)
	
	canvasLayout_NC = QVBoxLayout()
	canvasLayout_NC.addWidget(canvas_NC)
	
	canvasLayout_AOC = QVBoxLayout()
	canvasLayout_AOC.addWidget(canvas_AOC)	
	#### ------------------------------------------

	def __init__(self):
		super().__init__()
		self.initUI()
		self.initRunDataModel()

		self.setLayout(self.layout)
		self.setGeometry(300, 200, 1300, 800)

	def initUI(self):
		self.setWindowTitle("Simulator")
		self.setWindowIcon(QIcon('Save.png'))
				
		self.layout = QHBoxLayout()
		fontSize = 12
		fontShape = "Arial"
		
		#### Define tags ------------------------------------------		
		self.Month_tag_CC = ['January(1)','February(2)','March(3)','April(4)','May(5)','June(6)', \
								'July(7)','August(8)','Setember(9)','October(10)','November(11)','December(12)']
		MainWindow.Temp_tag_CC = [ i for i in np.arange(10, 30.5, 0.5) ]
		self.Time_tag_CC = [ i for i in np.arange(0, 25, 1) ]
		
		MainWindow.Time_tag_AOC = [ i for i in np.arange(0, 25, 1) ]		
		MainWindow.Temp_tag_AOC = [ i for i in np.arange(22, 28, 0.5) ]
		MainWindow.TolTemp_tag_AOC = [round(i,2) for i in np.arange(0, 2, 0.05) ]
		MainWindow.Month_tag_AOC = ['January(1)','February(2)','March(3)','April(4)','May(5)','June(6)', \
								'July(7)','August(8)','Setember(9)','October(10)','November(11)','December(12)']
		
		MainWindow.Month_tag_NC = ['January(1)','February(2)','March(3)','April(4)','May(5)','June(6)', \
								'July(7)','August(8)','Setember(9)','October(10)','November(11)','December(12)']
		MainWindow.Temp_tag_NC = [i for i in np.arange(3, 5.5, 0.5)]
		MainWindow.Tol_tag_NC = [round(i,2) for i in np.arange(0.2, 0.55, 0.05)]
		MainWindow.Time_tag_NC = [ i for i in np.arange(0, 25, 1) ]
		
		#### ------------------------------------------
		
		## Initialize tab screen
		## tab: load forecasting
		tabs = QTabWidget()
		
		#### Add tabs ------------------------------------------
		tab_SM = QWidget()
		tabs.addTab(tab_SM, "Summary")
		tab_CC = QWidget()
		tabs.addTab(tab_CC, "Cycle Control")
		tab_EC = QWidget()
		tabs.addTab(tab_EC, "Enthalpy Control")
		tab_NC = QWidget()
		tabs.addTab(tab_NC, "Night Purge Control")
		tab_AOC = QWidget()
		tabs.addTab(tab_AOC, "AHU Optimul Control")
		# Add tabs to widget
		tabs.setFont(QFont(fontShape, fontSize))
		self.layout.addWidget(tabs)
		#### ------------------------------------------
		
		#### Set layout (Cycle control) ------------------------------------------		
		## Update button
		updateButton_CC = QPushButton("  Update Model  ")
		updateButton_CC.setFont(QFont(fontShape, fontSize))
		updateButton_CC.clicked.connect(self.ClickedUpdateData_CC)
						
		## GroupBox
		groupBox_CC = QGroupBox("User Config.", self)
		lbx_config_CC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBox_CC.setLayout(lbx_config_CC)
		
		## Select AHU model
		AHUNum_CC = QLabel()
		AHUNum_CC.setText("AHU Model : ")
		AHUNum_CC.setFont(QFont(fontShape, fontSize))
		self.Mode_Combo_CC = QComboBox(self)
		for i in range(len(MainWindow.AHU_tag)):
			self.Mode_Combo_CC.addItem("  " + MainWindow.AHU_tag[i] + "  ")
		self.Mode_Combo_CC.activated.connect(self.ChangeAHUNUm_CC)
		self.Mode_Combo_CC.setCurrentIndex(0)
						
		## Cooling season boundary
		# initial month
		groupBoxCoolMonth_CC = QGroupBox("Cooling Boundary", self)
		groupBoxCoolMonth_CC.setFont(QFont(fontShape, fontSize))
		lbx_CoolMonth_CC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBoxCoolMonth_CC.setLayout(lbx_CoolMonth_CC)
		InitialCoolMonth_CC = QLabel()
		InitialCoolMonth_CC.setText("Initial month : ")
		InitialCoolMonth_CC.setFont(QFont(fontShape, fontSize))
		self.InitialCoolMonth_Combo_CC = QComboBox(self)
		for i in range(len(self.Month_tag_CC)):
			self.InitialCoolMonth_Combo_CC.addItem("  " + self.Month_tag_CC[i])
		self.InitialCoolMonth_Combo_CC.activated.connect(self.ChangeCoolMonthBoundary_CC)
		self.InitialCoolMonth_Combo_CC.setCurrentIndex(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['InitialCoolMonth'] - 1)
		lbx_CoolMonth_CC.addWidget(InitialCoolMonth_CC)
		lbx_CoolMonth_CC.addWidget(self.InitialCoolMonth_Combo_CC)		
		# final month
		FinalCoolMonth_CC = QLabel()
		FinalCoolMonth_CC.setText("Final month : ")
		FinalCoolMonth_CC.setFont(QFont(fontShape, fontSize))
		self.FinalCoolMonth_Combo_CC = QComboBox(self)		
		for i in range(len(self.Month_tag_CC)):
			self.FinalCoolMonth_Combo_CC.addItem("  " + self.Month_tag_CC[i])
		self.FinalCoolMonth_Combo_CC.activated.connect(self.ChangeCoolMonthBoundary_CC)
		self.FinalCoolMonth_Combo_CC.setCurrentIndex(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['FinalCoolMonth'] - 1)
		lbx_CoolMonth_CC.addWidget(FinalCoolMonth_CC)
		lbx_CoolMonth_CC.addWidget(self.FinalCoolMonth_Combo_CC)
		
		## Time boundary
		# initial time
		groupBoxTime_CC = QGroupBox("Time Boundary", self)
		groupBoxTime_CC.setFont(QFont(fontShape, fontSize))
		lbx_Time_CC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBoxTime_CC.setLayout(lbx_Time_CC)
		InitialTime_CC = QLabel()
		InitialTime_CC.setText("Initial time : ")
		InitialTime_CC.setFont(QFont(fontShape, fontSize))
		self.InitialTime_Combo_CC = QComboBox(self)		
		for i in range(len(self.Time_tag_CC)):
			self.InitialTime_Combo_CC.addItem("  " + str(self.Time_tag_CC[i]) + ":00")
		self.InitialTime_Combo_CC.activated.connect(self.ChangeTimeBoundary_CC)
		self.InitialTime_Combo_CC.setCurrentIndex(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['InitialTime'])
		lbx_Time_CC.addWidget(InitialTime_CC)
		lbx_Time_CC.addWidget(self.InitialTime_Combo_CC)		
		# final time
		FinalTime_CC = QLabel()
		FinalTime_CC.setText("Final time : ")
		FinalTime_CC.setFont(QFont(fontShape, fontSize))
		self.FinalTime_Combo_CC = QComboBox(self)		
		for i in range(len(self.Time_tag_CC)):
			self.FinalTime_Combo_CC.addItem("  " + str(self.Time_tag_CC[i]) + ":00")
		self.FinalTime_Combo_CC.activated.connect(self.ChangeTimeBoundary_CC)
		self.FinalTime_Combo_CC.setCurrentIndex(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['FinalTime'])
		lbx_Time_CC.addWidget(FinalTime_CC)
		lbx_Time_CC.addWidget(self.FinalTime_Combo_CC)
		
		## Temperature boundary
		groupBoxTemp_CC = QGroupBox("Temperature Boundary", self)
		groupBoxTemp_CC.setFont(QFont(fontShape, fontSize))
		lbx_Temp_CC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBoxTemp_CC.setLayout(lbx_Temp_CC)		
		# upper limit
		TargetTemp_CC = QLabel()
		TargetTemp_CC.setText("Upper temp. : ")
		TargetTemp_CC.setFont(QFont(fontShape, fontSize))
		self.UpperTemp_Combo_CC = QComboBox(self)
		for i in range(len(MainWindow.Temp_tag_CC)):
			self.UpperTemp_Combo_CC.addItem("  " + str(MainWindow.Temp_tag_CC[i]) + " ℃")
		self.UpperTemp_Combo_CC.activated.connect(self.ChangeTempBoundary_CC)
		self.UpperTemp_Combo_CC.setCurrentIndex(MainWindow.Temp_tag_CC.index(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['UpperTemp']))
		lbx_Temp_CC.addWidget(TargetTemp_CC)
		lbx_Temp_CC.addWidget(self.UpperTemp_Combo_CC)
		# lower limt
		LowerTemp_CC = QLabel()
		LowerTemp_CC.setText("Lower temp. : ")
		LowerTemp_CC.setFont(QFont(fontShape, fontSize))
		self.LowerTemp_Combo_CC = QComboBox(self)
		for i in range(len(MainWindow.Temp_tag_CC)):
			self.LowerTemp_Combo_CC.addItem("  " + str(MainWindow.Temp_tag_CC[i]) + " ℃")
		self.LowerTemp_Combo_CC.activated.connect(self.ChangeTempBoundary_CC)
		self.LowerTemp_Combo_CC.setCurrentIndex(MainWindow.Temp_tag_CC.index(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['LowerTemp']))
		lbx_Temp_CC.addWidget(LowerTemp_CC)		
		lbx_Temp_CC.addWidget(self.LowerTemp_Combo_CC)		
					
		## Control Schedule
		groupBoxControlSC_CC = QGroupBox("Controlled AHU On/Off (+#)", self)
		groupBoxControlSC_CC.setFont(QFont(fontShape, fontSize))
		lbx_ControlSC_CC = QBoxLayout(QBoxLayout.LeftToRight, parent=self)
		groupBoxControlSC_CC.setLayout(lbx_ControlSC_CC)		
		# User Control Schedule	
		self.checkBoxSchedule1_CC = QCheckBox('(1)', self)
		self.checkBoxSchedule1_CC.toggle()
		self.checkBoxSchedule1_CC.stateChanged.connect(self.changeSchedule1_CC)
		self.checkBoxSchedule2_CC = QCheckBox('(2)', self)
		self.checkBoxSchedule2_CC.toggle()
		self.checkBoxSchedule2_CC.stateChanged.connect(self.changeSchedule2_CC)
		self.checkBoxSchedule3_CC = QCheckBox('(3)', self)
		self.checkBoxSchedule3_CC.toggle()
		self.checkBoxSchedule3_CC.stateChanged.connect(self.changeSchedule3_CC)
		self.checkBoxSchedule4_CC = QCheckBox('(4)', self)
		self.checkBoxSchedule4_CC.toggle()
		self.checkBoxSchedule4_CC.stateChanged.connect(self.changeSchedule4_CC)
		lbx_ControlSC_CC.addWidget(self.checkBoxSchedule1_CC)
		lbx_ControlSC_CC.addWidget(self.checkBoxSchedule2_CC)
		lbx_ControlSC_CC.addWidget(self.checkBoxSchedule3_CC)
		lbx_ControlSC_CC.addWidget(self.checkBoxSchedule4_CC)
		if MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][0] == 1:
			self.checkBoxSchedule1_CC.setChecked(True)
		else:
			self.checkBoxSchedule1_CC.setChecked(False)
		if MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][1] == 1:
			self.checkBoxSchedule2_CC.setChecked(True)
		else:
			self.checkBoxSchedule2_CC.setChecked(False)
		if MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][2] == 1:
			self.checkBoxSchedule3_CC.setChecked(True)
		else:
			self.checkBoxSchedule3_CC.setChecked(False)
		if MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][3] == 1:
			self.checkBoxSchedule4_CC.setChecked(True)
		else:
			self.checkBoxSchedule4_CC.setChecked(False)
			
		## Config GroupBox layout
		lbx_config_CC.addWidget(AHUNum_CC)
		lbx_config_CC.addWidget(self.Mode_Combo_CC)
		lbx_config_CC.addWidget(groupBoxCoolMonth_CC)
		lbx_config_CC.addWidget(groupBoxTime_CC)
		lbx_config_CC.addWidget(groupBoxTemp_CC)
		lbx_config_CC.addWidget(groupBoxControlSC_CC)

		# canvas Layout
		btnLayout_CC = QVBoxLayout()
		btnLayout_CC.addWidget(QLabel())
		btnLayout_CC.addWidget(updateButton_CC)	
		btnLayout_CC.addWidget(QLabel())
		btnLayout_CC.addWidget(groupBox_CC)
		btnLayout_CC.addStretch(1)

		tab_CC.layout = QGridLayout(self)
		tab_CC.layout.addLayout(MainWindow.canvasLayout_CC, 0, 0)
		tab_CC.layout.addLayout(btnLayout_CC, 0, 1)
		tab_CC.setLayout(tab_CC.layout)
		#### ------------------------------------------
		
		#### Set layout (Enthalpy Control Simulator) ------------------------------------------
		updateButton_EC = QPushButton("  Update Model  ")
		updateButton_EC.clicked.connect(self.ClickedUpdateData_EC)
		updateButton_EC.setFont(QFont(fontShape, fontSize))

		## GroupBox
		groupBox_EC = QGroupBox("User config.", self)
		lbx_config_EC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBox_EC.setLayout(lbx_config_EC)
		
		# AHU label
		AHUNum_tab_EC = QLabel()
		AHUNum_tab_EC.setText("AHU Model : ")
		AHUNum_tab_EC.setFont(QFont(fontShape, fontSize))
		
		# AHU model combobox
		self.Mode_Combo_EC = QComboBox(self)
		for i in range(len(MainWindow.AHU_tag)):
			self.Mode_Combo_EC.addItem("  " + MainWindow.AHU_tag[i] + "  ")
		## connect function
		self.Mode_Combo_EC.activated.connect(self.ChangeAHUNUm_EC)
		self.Mode_Combo_EC.setCurrentIndex(0)

		## text 
		MainWindow.Saving_tab_EC = QLabel()
		MainWindow.Saving_tab_EC.setAlignment(Qt.AlignVCenter)
		MainWindow.Saving_tab_EC.setFont(QFont(fontShape, fontSize, QFont.Bold))
		MainWindow.Saving_tab_EC.setStyleSheet("Color : lightcoral")
		
		textLayout_EC = QVBoxLayout()
		textLayout_EC.addStretch(2)
		textLayout_EC.addWidget(MainWindow.Saving_tab_EC)

		# Config GroupBox layout
		lbx_config_EC.addWidget(AHUNum_tab_EC)
		lbx_config_EC.addWidget(self.Mode_Combo_EC)
		# lbx_config_EC.addWidget(groupBoxTrain_EC)

		# canvas Layout
		btnLayout_EC = QVBoxLayout()
		btnLayout_EC.addWidget(QLabel())
		btnLayout_EC.addWidget(updateButton_EC)
		btnLayout_EC.addWidget(QLabel())
		btnLayout_EC.addWidget(groupBox_EC)
		btnLayout_EC.addStretch(1)
	
		tab_EC.layout = QGridLayout(self)	
		tab_EC.layout.addLayout(textLayout_EC, 0, 0)
		tab_EC.layout.addLayout(MainWindow.canvasLayout_EC, 1, 0)
		tab_EC.layout.addLayout(btnLayout_EC, 1, 1)
		tab_EC.setLayout(tab_EC.layout)		
		#### ------------------------------------------

		#### Set layout (Night Purge Control Simulator) ---------------------------------------
		## Update button
		updateButton_NC = QPushButton("  Update Model  ")
		updateButton_NC.clicked.connect(self.ClickedUpdateData_NC)
		updateButton_NC.setFont(QFont(fontShape, fontSize))

		## GroupBox
		groupBox_NC = QGroupBox("User config.", self)
		lbx_config_NC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBox_NC.setLayout(lbx_config_NC)

		# AHU label
		AHUNum_tab_NC = QLabel()
		AHUNum_tab_NC.setText("AHU Model : ")
		AHUNum_tab_NC.setFont(QFont(fontShape, fontSize))

		# AHU model combobox
		self.Mode_Combo_NC = QComboBox(self)
		for i in range(len(MainWindow.AHU_tag)):
			self.Mode_Combo_NC.addItem("  " + MainWindow.AHU_tag[i] + "  ")
		## connect function
		self.Mode_Combo_NC.activated.connect(self.ChangeAHUNum_NC)
		self.Mode_Combo_NC.setCurrentIndex(0)

		# Cooling boundary
		# initial month
		groupBoxCoolMonth_NC = QGroupBox("Cooling Boundary", self)
		groupBoxCoolMonth_NC.setFont(QFont(fontShape, fontSize))
		lbx_CoolMonth_NC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBoxCoolMonth_NC.setLayout(lbx_CoolMonth_NC)
		InitialCoolMonth_NC = QLabel()
		InitialCoolMonth_NC.setText("Initial month : ")
		InitialCoolMonth_NC.setFont(QFont(fontShape, fontSize))
		self.InitialCoolMonth_Combo_NC = QComboBox(self)
		for i in range(len(self.Month_tag_NC)):
			self.InitialCoolMonth_Combo_NC.addItem("  " + self.Month_tag_NC[i])
		self.InitialCoolMonth_Combo_NC.activated.connect(self.ChangeCoolMonthBoundary_NC)
		self.InitialCoolMonth_Combo_NC.setCurrentIndex(
			MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['InitialCoolMonth'] - 1)
		lbx_CoolMonth_NC.addWidget(InitialCoolMonth_NC)
		lbx_CoolMonth_NC.addWidget(self.InitialCoolMonth_Combo_NC)
		# final month
		FinalCoolMonth_NC = QLabel()
		FinalCoolMonth_NC.setText("Final month : ")
		FinalCoolMonth_NC.setFont(QFont(fontShape, fontSize))
		self.FinalCoolMonth_Combo_NC = QComboBox(self)
		for i in range(len(self.Month_tag_NC)):
			self.FinalCoolMonth_Combo_NC.addItem("  " + self.Month_tag_NC[i])
		self.FinalCoolMonth_Combo_NC.activated.connect(self.ChangeCoolMonthBoundary_NC)
		self.FinalCoolMonth_Combo_NC.setCurrentIndex(
			MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['FinalCoolMonth'] - 1)
		lbx_CoolMonth_NC.addWidget(FinalCoolMonth_NC)
		lbx_CoolMonth_NC.addWidget(self.FinalCoolMonth_Combo_NC)

		# Time Boundary
		# # initial time
		groupBoxTime_NC = QGroupBox("Time Boundary", self)
		groupBoxTime_NC.setFont(QFont(fontShape, fontSize))
		lbx_Time_NC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBoxTime_NC.setLayout(lbx_Time_NC)
		# final time
		FinalTime_NC = QLabel()
		FinalTime_NC.setText("Final time : ")
		FinalTime_NC.setFont(QFont(fontShape, fontSize))
		self.FinalTime_Combo_NC = QComboBox(self)
		for i in range(len(self.Time_tag_NC)):
			self.FinalTime_Combo_NC.addItem("  " + str(self.Time_tag_NC[i]) + ":00")
		self.FinalTime_Combo_NC.activated.connect(self.ChangeTimeBoundary_NC)
		self.FinalTime_Combo_NC.setCurrentIndex(MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['end_hour'])
		lbx_Time_NC.addWidget(FinalTime_NC)
		lbx_Time_NC.addWidget(self.FinalTime_Combo_NC)

		## setting temp boundary
		groupBoxTemp_NC = QGroupBox("Temp. Boundary", self)
		groupBoxTemp_NC.setFont(QFont(fontShape, fontSize))
		lbx_Temp_NC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBoxTemp_NC.setLayout(lbx_Temp_NC)
		# setting RAOA
		TargetTemp_NC = QLabel()
		TargetTemp_NC.setText("Difference (RA/OA) : ")
		TargetTemp_NC.setFont(QFont(fontShape, fontSize))
		self.TempDifference_Combo_NC = QComboBox(self)
		for i in range(len(MainWindow.Temp_tag_NC)):
			self.TempDifference_Combo_NC.addItem("  " + str(MainWindow.Temp_tag_NC[i]) + " ℃")
		self.TempDifference_Combo_NC.activated.connect(self.ChangeTempDifference_NC)
		self.TempDifference_Combo_NC.setCurrentIndex(
			MainWindow.Temp_tag_NC.index(MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['delta_RAOA']))
		lbx_Temp_NC.addWidget(TargetTemp_NC)
		lbx_Temp_NC.addWidget(self.TempDifference_Combo_NC)

		# setting temp. tolerance
		TempTolerance_NC = QLabel()
		TempTolerance_NC.setText("Tolerance : ")
		TempTolerance_NC.setFont(QFont(fontShape, fontSize))
		self.TempTolerance_Combo_NC = QComboBox(self)
		for i in range(len(MainWindow.Tol_tag_NC)):
			self.TempTolerance_Combo_NC.addItem("  " + str(MainWindow.Tol_tag_NC[i]) + " ℃")
		self.TempTolerance_Combo_NC.activated.connect(self.ChangeTempTolerance_NC)
		self.TempTolerance_Combo_NC.setCurrentIndex(
			MainWindow.Tol_tag_NC.index(MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['margin_error']))
		lbx_Temp_NC.addWidget(TempTolerance_NC)
		lbx_Temp_NC.addWidget(self.TempTolerance_Combo_NC)

		# text
		MainWindow.NC_able_tab = QLabel()
		MainWindow.NC_able_tab.setAlignment(Qt.AlignVCenter)
		MainWindow.NC_able_tab.setFont(QFont(fontShape, fontSize, QFont.Bold))
		MainWindow.NC_able_tab.setStyleSheet("Color : lightcoral")

		MainWindow.NC_saving_tab = QLabel()
		MainWindow.NC_saving_tab.setAlignment(Qt.AlignVCenter)
		MainWindow.NC_saving_tab.setFont(QFont(fontShape, fontSize, QFont.Bold))
		MainWindow.NC_saving_tab.setStyleSheet("Color : lightcoral")

		textLayout_NC = QVBoxLayout()
		textLayout_NC.addStretch(2)
		textLayout_NC.addWidget(MainWindow.NC_able_tab)
		textLayout_NC.addWidget(MainWindow.NC_saving_tab)

		# Config GroupBox layout
		lbx_config_NC.addWidget(AHUNum_tab_NC)
		lbx_config_NC.addWidget(self.Mode_Combo_NC)
		lbx_config_NC.addWidget(groupBoxCoolMonth_NC)
		lbx_config_NC.addWidget(groupBoxTime_NC)
		lbx_config_NC.addWidget(groupBoxTemp_NC)
		# lbx_config_NC.addWidget(groupBoxTol_NC)

		# canvas Layout
		btnLayout_NC = QVBoxLayout()
		btnLayout_NC.addWidget(QLabel())
		btnLayout_NC.addWidget(updateButton_NC)
		btnLayout_NC.addWidget(QLabel())
		btnLayout_NC.addWidget(groupBox_NC)
		# btnLayout_NC.addWidget(groupBoxRAOA_NC)
		# btnLayout_NC.addWidget(groupBoxTol_NC)
		btnLayout_NC.addStretch(1)

		tab_NC.layout = QGridLayout(self)
		tab_NC.layout.addLayout(textLayout_NC, 0, 0)
		tab_NC.layout.addLayout(MainWindow.canvasLayout_NC, 1, 0)
		tab_NC.layout.addLayout(btnLayout_NC, 1, 1)
		tab_NC.setLayout(tab_NC.layout)
		#### ------------------------------------------
		
		#### Set layout (Cooler/Heater-AHU optimul control) ------------------------------------------
		updateButton_AOC = QPushButton("  Update Model  ")
		# changing font and size of text 
		updateButton_AOC.setFont(QFont(fontShape, fontSize))
		updateButton_AOC.clicked.connect(self.ClickedUpdateData_AOC)
		
		## GroupBox
		groupBox_AOC = QGroupBox("User config.", self)
		lbx_config_AOC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBox_AOC.setLayout(lbx_config_AOC)
		
		# AHU label
		AHUNum_tab_AOC = QLabel()
		AHUNum_tab_AOC.setText("AHU Model : ")
		AHUNum_tab_AOC.setFont(QFont(fontShape, fontSize))
		# AHU model combobox
		self.AHUNum_Combo_AOC = QComboBox(self)
		for i in range(len(MainWindow.AHU_tag)):
			self.AHUNum_Combo_AOC.addItem("  " + MainWindow.AHU_tag[i] + "  ")
		## connect function
		self.AHUNum_Combo_AOC.activated.connect(self.ChangeAHUNum_AOC)
		self.AHUNum_Combo_AOC.setCurrentIndex(0)
		MainWindow.AHUidx_AOC = self.AHUNum_Combo_AOC.currentIndex()
		
		# GroupBox (Start/Stop mode)
		groupBoxMode_AOC = QGroupBox("Control mode", self)
		groupBoxMode_AOC.setFont(QFont(fontShape, fontSize))
		lbx_Mode_AOC = QBoxLayout(QBoxLayout.LeftToRight, parent=self)
		groupBoxMode_AOC.setLayout(lbx_Mode_AOC)
		MainWindow.radioStart_AOC = QRadioButton("Start", self)
		MainWindow.radioStop_AOC = QRadioButton("Stop", self)		
		MainWindow.radioStart_AOC.clicked.connect(self.radioButtonClickedMode_AOC)
		MainWindow.radioStop_AOC.clicked.connect(self.radioButtonClickedMode_AOC)		
		MainWindow.radioStart_AOC.setChecked(True) if MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['SSRadioBtn'] == 'Start' else MainWindow.radioStop_AOC.setChecked(True)
		lbx_Mode_AOC.addWidget(MainWindow.radioStart_AOC)
		lbx_Mode_AOC.addWidget(MainWindow.radioStop_AOC)

		## Cooling season boundary
		# initial month
		groupBoxMonth_AOC = QGroupBox("Training boundary", self)
		groupBoxMonth_AOC.setFont(QFont(fontShape, fontSize))
		lbx_Month_AOC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBoxMonth_AOC.setLayout(lbx_Month_AOC)
		InitialMonth_AOC = QLabel()
		InitialMonth_AOC.setText("Initial month : ")
		InitialMonth_AOC.setFont(QFont(fontShape, fontSize))
		self.InitialMonth_Combo_AOC = QComboBox(self)
		for i in range(len(self.Month_tag_AOC)):
			self.InitialMonth_Combo_AOC.addItem("  " + self.Month_tag_AOC[i])
		self.InitialMonth_Combo_AOC.activated.connect(self.ChangeMonthBoundary_AOC)
		self.InitialMonth_Combo_AOC.setCurrentIndex(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['InitialCoolMonth'] - 1)
		lbx_Month_AOC.addWidget(InitialMonth_AOC)
		lbx_Month_AOC.addWidget(self.InitialMonth_Combo_AOC)		
		# final month
		FinalMonth_AOC = QLabel()
		FinalMonth_AOC.setText("Final month : ")
		FinalMonth_AOC.setFont(QFont(fontShape, fontSize))
		self.FinalMonth_Combo_AOC = QComboBox(self)		
		for i in range(len(self.Month_tag_AOC)):
			self.FinalMonth_Combo_AOC.addItem("  " + self.Month_tag_AOC[i])
		self.FinalMonth_Combo_AOC.activated.connect(self.ChangeMonthBoundary_AOC)
		self.FinalMonth_Combo_AOC.setCurrentIndex(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['FinalCoolMonth'] - 1)
		lbx_Month_AOC.addWidget(FinalMonth_AOC)
		lbx_Month_AOC.addWidget(self.FinalMonth_Combo_AOC)
		
		## Time boundary
		# initial time
		groupBoxTime_AOC = QGroupBox("Time boundary", self)
		groupBoxTime_AOC.setFont(QFont(fontShape, fontSize))
		lbx_Time_AOC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBoxTime_AOC.setLayout(lbx_Time_AOC)
		OpeningTime_AOC = QLabel()
		OpeningTime_AOC.setText("Opening time : ")
		OpeningTime_AOC.setFont(QFont(fontShape, fontSize))
		self.OpeningTime_Combo_AOC = QComboBox(self)
		for i in range(len(MainWindow.Time_tag_AOC)):
			self.OpeningTime_Combo_AOC.addItem("  " + str(MainWindow.Time_tag_AOC[i]) + ":00")
		self.OpeningTime_Combo_AOC.activated.connect(self.ChangeTimeBoundary_AOC)
		self.OpeningTime_Combo_AOC.setCurrentIndex(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['OpeningTime'])
		lbx_Time_AOC.addWidget(OpeningTime_AOC)
		lbx_Time_AOC.addWidget(self.OpeningTime_Combo_AOC)		
		# final time
		ClosingTime_AOC = QLabel()
		ClosingTime_AOC.setText("Closing time : ")
		ClosingTime_AOC.setFont(QFont(fontShape, fontSize))
		self.ClosingTime_Combo_AOC = QComboBox(self)		
		for i in range(len(MainWindow.Time_tag_AOC)):
			self.ClosingTime_Combo_AOC.addItem("  " + str(MainWindow.Time_tag_AOC[i]) + ":00")
		self.ClosingTime_Combo_AOC.activated.connect(self.ChangeTimeBoundary_AOC)
		self.ClosingTime_Combo_AOC.setCurrentIndex(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['ClosingTime'])
		lbx_Time_AOC.addWidget(ClosingTime_AOC)
		lbx_Time_AOC.addWidget(self.ClosingTime_Combo_AOC)
		
		# Target temperature
		TargetTempAHU_AOC = QLabel()
		TargetTempAHU_AOC.setText("Setting temp. (RA) : ")
		TargetTempAHU_AOC.setFont(QFont(fontShape, fontSize))
		self.TargetTempAHU_Combo_AOC = QComboBox(self)			
		for i in range(len(MainWindow.Temp_tag_AOC)):
			self.TargetTempAHU_Combo_AOC.addItem("  " + str(MainWindow.Temp_tag_AOC[i]) + " ℃")
		self.TargetTempAHU_Combo_AOC.activated.connect(self.ChangeAHUTargetTemp_AOC)
		self.TargetTempAHU_Combo_AOC.setCurrentIndex(MainWindow.Temp_tag_AOC.index(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['AHU_TargetTemp']))
			
		# Tolerance tempearture
		ToleranceTemp_AOC = QLabel()
		ToleranceTemp_AOC.setText("Tolerance temp. (RA) : ")
		ToleranceTemp_AOC.setFont(QFont(fontShape, fontSize))
		self.ToleranceTemp_Combo_AOC = QComboBox(self)		
		for i in range(len(MainWindow.TolTemp_tag_AOC)):
			self.ToleranceTemp_Combo_AOC.addItem("  " + str(MainWindow.TolTemp_tag_AOC[i]) + " ℃")
		self.ToleranceTemp_Combo_AOC.activated.connect(self.ChangeToleranceTime_AOC)
		self.ToleranceTemp_Combo_AOC.setCurrentIndex(MainWindow.TolTemp_tag_AOC.index(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['AHU_TempTolerance']))
		
		
		## Config GroupBox layout
		lbx_config_AOC.addWidget(AHUNum_tab_AOC)
		lbx_config_AOC.addWidget(self.AHUNum_Combo_AOC)
		lbx_config_AOC.addWidget(groupBoxMode_AOC)		
		lbx_config_AOC.addWidget(groupBoxMonth_AOC)
		lbx_config_AOC.addWidget(groupBoxTime_AOC)
		lbx_config_AOC.addWidget(TargetTempAHU_AOC)
		lbx_config_AOC.addWidget(self.TargetTempAHU_Combo_AOC)
		lbx_config_AOC.addWidget(ToleranceTemp_AOC)
		lbx_config_AOC.addWidget(self.ToleranceTemp_Combo_AOC)
		
		# canvas Layout
		btnLayout_AOC = QVBoxLayout()
		btnLayout_AOC.addWidget(QLabel())
		btnLayout_AOC.addWidget(updateButton_AOC)	
		btnLayout_AOC.addWidget(QLabel())
		btnLayout_AOC.addWidget(groupBox_AOC)
		btnLayout_AOC.addStretch(1)

		tab_AOC.layout = QGridLayout(self)
		tab_AOC.layout.addLayout(MainWindow.canvasLayout_AOC, 0, 0)
		tab_AOC.layout.addLayout(btnLayout_AOC, 0, 1)
		tab_AOC.setLayout(tab_AOC.layout)
		#### ------------------------------------------		
		
		
		#### Set layout (Summary) ------------------------------------------		
		## GroupBox summary (Cylce Control Control)
		groupBox_SM_CC = QGroupBox("Cycle Control", self)
		lbx_config_SM_CC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBox_SM_CC.setLayout(lbx_config_SM_CC)		
		## GroupBox layout summary 
		MainWindow.AHUNum_SM_CC = []
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUNum_SM_CC.append(QLabel())
			MainWindow.AHUNum_SM_CC[i].setText("[" + MainWindow.AHU_tag[i] + "] Unknown")
			lbx_config_SM_CC.addWidget(MainWindow.AHUNum_SM_CC[i])
		
		## GroupBox summary (Enthalpy Control)
		groupBox_SM_EC = QGroupBox("Enthalpy Control", self)
		lbx_config_SM_EC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBox_SM_EC.setLayout(lbx_config_SM_EC)
		MainWindow.AHUNum_SM_EC = []
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUNum_SM_EC.append(QLabel())
			MainWindow.AHUNum_SM_EC[i].setText("[" + MainWindow.AHU_tag[i] + "] < : Unknown, Unknown, Unknown, Unknown [%]>"
											   + "<Energy Savings : Unknown [Wh/kg]> ")
			lbx_config_SM_EC.addWidget(MainWindow.AHUNum_SM_EC[i])
			
		## GroupBox summary (Night Purge Control)
		groupBox_SM_NC = QGroupBox("Night Purge Control", self)
		lbx_config_SM_NC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBox_SM_NC.setLayout(lbx_config_SM_NC)
		MainWindow.AHUNum_SM_NC = []
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUNum_SM_NC.append(QLabel())
			MainWindow.AHUNum_SM_NC[i].setText("[" + MainWindow.AHU_tag[i] + "] Unknown ~ Unknown ")
			lbx_config_SM_NC.addWidget(MainWindow.AHUNum_SM_NC[i])

		## GroupBox summary (Cooler/Heater-AHU Optimul Control)
		groupBox_SM_AOC = QGroupBox("AHU Optimul Control", self)
		lbx_config_SM_AOC = QBoxLayout(QBoxLayout.TopToBottom, parent=self)
		groupBox_SM_AOC.setLayout(lbx_config_SM_AOC)
		MainWindow.AHUNum_SM_AOC = []
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUNum_SM_AOC.append(QLabel())
			MainWindow.AHUNum_SM_AOC[i].setText("[" + MainWindow.AHU_tag[i] + "] Start: Unknown, Stop: Unknown")
			lbx_config_SM_AOC.addWidget(MainWindow.AHUNum_SM_AOC[i])
		
		# Text Layout summary
		txtLayout_SM_Left = QVBoxLayout()
		txtLayout_SM_Left.addWidget(groupBox_SM_EC)
		txtLayout_SM_Right = QVBoxLayout()
		txtLayout_SM_Left.addWidget(groupBox_SM_AOC)
		txtLayout_SM_Right.addWidget(groupBox_SM_CC)
		txtLayout_SM_Right.addWidget(groupBox_SM_NC)
	
		tab_SM.layout = QGridLayout(self)
		tab_SM.layout.addLayout(txtLayout_SM_Left, 0, 0)
		tab_SM.layout.addLayout(txtLayout_SM_Right, 0, 1)
		tab_SM.setLayout(tab_SM.layout)
		#### ------------------------------------------
		
	def initRunDataModel(self):	
		#### Cycle Control Tab ------------------------------------------	
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUidx_CC = i
			MainWindow.ModuleExe(MainWindow, 'CC')
			if i == 0:
				MainWindow.UpdateFigure(MainWindow, 'CC')
		MainWindow.AHUidx_CC = 0
		#### ------------------------------------------	
		
		#### AHU Optimal Control Tab ------------------------------------------
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUidx_AOC = i
			MainWindow.ModuleExe(MainWindow, 'AOC')
			if i == 0:
				MainWindow.UpdateFigure(MainWindow, 'AOC')
		MainWindow.AHUidx_AOC = 0
		#### ------------------------------------------	
		
		#### Enthalpy Control Tab ------------------------------------------
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUidx_EC = i
			MainWindow.ModuleExe(MainWindow, 'EC')
			if i == 0:
				MainWindow.UpdateFigure(MainWindow, 'EC')
		MainWindow.AHUidx_EC = 0
		#### ------------------------------------------
			
		#### Night Purge Control Tab ---------------------------------------
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUidx_NC = i
			MainWindow.ModuleExe(MainWindow, 'NC')
			if i == 0:
				MainWindow.UpdateFigure(MainWindow, 'NC')
		MainWindow.AHUidx_NC = 0
		#### ------------------------------------------
		
				
	def ChangeAHUNUm_CC(self):
		MainWindow.AHUidx_CC = int(self.Mode_Combo_CC.currentIndex())
		## Load and show the subparameters of the selected AHU model
		self.InitialCoolMonth_Combo_CC.setCurrentIndex(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['InitialCoolMonth'] - 1)
		self.FinalCoolMonth_Combo_CC.setCurrentIndex(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['FinalCoolMonth'] - 1)		
		self.InitialTime_Combo_CC.setCurrentIndex(self.Time_tag_CC.index(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['InitialTime']))
		self.FinalTime_Combo_CC.setCurrentIndex(self.Time_tag_CC.index(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['FinalTime']))
		self.LowerTemp_Combo_CC.setCurrentIndex(MainWindow.Temp_tag_CC.index(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['LowerTemp']))
		self.UpperTemp_Combo_CC.setCurrentIndex(MainWindow.Temp_tag_CC.index(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['UpperTemp']))
		
		if MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][0] == 1:
			self.checkBoxSchedule1_CC.setChecked(True)
		else:
			self.checkBoxSchedule1_CC.setChecked(False)
		if MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][1] == 1:
			self.checkBoxSchedule2_CC.setChecked(True)
		else:
			self.checkBoxSchedule2_CC.setChecked(False)
		if MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][2] == 1:
			self.checkBoxSchedule3_CC.setChecked(True)
		else:
			self.checkBoxSchedule3_CC.setChecked(False)
		if MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][3] == 1:
			self.checkBoxSchedule4_CC.setChecked(True)
		else:
			self.checkBoxSchedule4_CC.setChecked(False)
		MainWindow.UpdateFigure(MainWindow, 'CC')
				
	def ChangeCoolMonthBoundary_CC(self):
		if float(self.FinalCoolMonth_Combo_CC.currentIndex()) <= float(self.InitialCoolMonth_Combo_CC.currentIndex()):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText("Month boundary is wrong")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			msgBox.exec()
			if self.InitialCoolMonth_Combo_CC.currentIndex() == len(self.Month_tag_CC) - 1:	# to check if InitialCoolMonth_Combo_CC is set to last index
				self.InitialCoolMonth_Combo_CC.setCurrentIndex(len(self.Month_tag_CC) - 2)
				self.FinalCoolMonth_Combo_CC.setCurrentIndex(len(self.Month_tag_CC) - 1)
			elif self.FinalCoolMonth_Combo_CC.currentIndex() == 0:	# to check if FinalCoolMonth_Combo_CC is set to first index
				self.InitialCoolMonth_Combo_CC.setCurrentIndex(0)
				self.FinalCoolMonth_Combo_CC.setCurrentIndex(1)
			else:
				self.FinalCoolMonth_Combo_CC.setCurrentIndex(self.InitialCoolMonth_Combo_CC.currentIndex() + 1)
		MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['InitialCoolMonth'] = self.InitialCoolMonth_Combo_CC.currentIndex() + 1
		MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['FinalCoolMonth'] = self.FinalCoolMonth_Combo_CC.currentIndex() + 1
	
	def ChangeTimeBoundary_CC(self):
		if float(self.FinalTime_Combo_CC.currentIndex()) <= float(self.InitialTime_Combo_CC.currentIndex()):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText("Time boundary is wrong")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			msgBox.exec()
			if self.InitialTime_Combo_CC.currentIndex() == len(self.Time_tag_CC) - 1:	# to check if InitialTime_Combo_CC is set to last index
				self.InitialTime_Combo_CC.setCurrentIndex(len(self.Time_tag_CC) - 2)
				self.FinalTime_Combo_CC.setCurrentIndex(len(self.Time_tag_CC) - 1)
			elif self.FinalTime_Combo_CC.currentIndex() == 0:	# to check if FinalTime_Combo_CC is set to first index
				self.InitialTime_Combo_CC.setCurrentIndex(0)
				self.FinalTime_Combo_CC.setCurrentIndex(1)
			else:
				self.FinalTime_Combo_CC.setCurrentIndex(self.InitialTime_Combo_CC.currentIndex() + 1)
		MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['InitialTime'] = int(self.InitialTime_Combo_CC.currentText().split(':')[0])	
		MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['FinalTime'] = int(self.FinalTime_Combo_CC.currentText().split(':')[0])
		
	def ChangeTempBoundary_CC(self):
		if float(self.UpperTemp_Combo_CC.currentIndex()) <= float(self.LowerTemp_Combo_CC.currentIndex()):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText("Temperature boundary is wrong")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				print('============ Popup Closed! ============')
			if self.LowerTemp_Combo_CC.currentIndex() == len(MainWindow.Temp_tag_CC) - 1:	# to check if LowerTemp_Combo_CC is set to last index
				self.LowerTemp_Combo_CC.setCurrentIndex(len(MainWindow.Temp_tag_CC) - 2)
				self.UpperTemp_Combo_CC.setCurrentIndex(len(MainWindow.Temp_tag_CC) - 1)
			elif self.UpperTemp_Combo_CC.currentIndex() == 0:	# to check if UpperTemp_Combo_CC is set to first index
				self.LowerTemp_Combo_CC.setCurrentIndex(0)
				self.UpperTemp_Combo_CC.setCurrentIndex(1)
			else:
				self.UpperTemp_Combo_CC.setCurrentIndex(self.LowerTemp_Combo_CC.currentIndex() + 1)
		MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['LowerTemp'] = float(self.LowerTemp_Combo_CC.currentText().split()[0])
		MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['UpperTemp'] = float(self.UpperTemp_Combo_CC.currentText().split()[0])		
		
	def changeSchedule1_CC(self, state):
		if state == Qt.Checked:
			MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][0] = 1  
			self.checkBoxSchedule1_CC.setChecked(True)
		else:
			MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][0] = 0	
			self.checkBoxSchedule1_CC.setChecked(False)
	def changeSchedule2_CC(self, state):
		if state == Qt.Checked:
			MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][1] = 1  
			self.checkBoxSchedule2_CC.setChecked(True)
		else:
			MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][1] = 0	
			self.checkBoxSchedule2_CC.setChecked(False)
	def changeSchedule3_CC(self, state):
		if state == Qt.Checked:
			MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][2] = 1  
			self.checkBoxSchedule3_CC.setChecked(True)
		else:
			MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][2] = 0	
			self.checkBoxSchedule3_CC.setChecked(False)
	def changeSchedule4_CC(self, state):
		if state == Qt.Checked:
			MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][3] = 1  
			self.checkBoxSchedule4_CC.setChecked(True)
		else:
			MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['CycleStatus'][3] = 0	
			self.checkBoxSchedule4_CC.setChecked(False)
			
		
	def ChangeAHUNUm_EC(self):
		MainWindow.AHUidx_EC = int(self.Mode_Combo_EC.currentIndex())
		MainWindow.UpdateFigure(MainWindow, 'EC')

	def ChangeTrainMonth_EC(self):
		MainWindow.InputParam_EC[MainWindow.AHUidx_EC]['Train_Month'] = self.TrainMonth_Combo_EC.currentIndex() + 1

	def ChangeTrainDay_EC(self):
		MainWindow.InputParam_EC[MainWindow.AHUidx_EC]['Train_Day'] = self.TrainDay_Combo_EC.currentIndex() + 1

	def ChangeAHUNum_NC(self):
		MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['AHUnum'] = int(self.Mode_Combo_NC.currentText().split('-')[1])
		MainWindow.AHUidx_NC = int(self.Mode_Combo_NC.currentIndex())
		## Load and show the subparameters of the selected AHU model
		# self.InitialTime_Combo_NC.setCurrentIndex(self.Time_tag_NC.index(MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['initial_time']))
		self.FinalTime_Combo_NC.setCurrentIndex(self.Time_tag_NC.index(MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['end_hour']))
		self.TempDifference_Combo_NC.setCurrentIndex(MainWindow.Temp_tag_NC.index(MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['delta_RAOA']))
		self.TempTolerance_Combo_NC.setCurrentIndex(MainWindow.Tol_tag_NC.index(MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['margin_error']))
		MainWindow.UpdateFigure(MainWindow, 'NC')

	def ChangeCoolMonthBoundary_NC(self):
		if float(self.FinalCoolMonth_Combo_NC.currentIndex()) <= float(self.InitialCoolMonth_Combo_NC.currentIndex()):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText("Month boundary is wrong")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			msgBox.exec()
			if self.InitialCoolMonth_Combo_NC.currentIndex() == len(self.Month_tag_NC) - 1:	# to check if InitialCoolMonth_Combo_CC is set to last index
				self.InitialCoolMonth_Combo_NC.setCurrentIndex(len(self.Month_tag_NC) - 2)
				self.FinalCoolMonth_Combo_NC.setCurrentIndex(len(self.Month_tag_NC) - 1)
			elif self.FinalCoolMonth_Combo_NC.currentIndex() == 0:	# to check if FinalCoolMonth_Combo_CC is set to first index
				self.InitialCoolMonth_Combo_NC.setCurrentIndex(0)
				self.FinalCoolMonth_Combo_NC.setCurrentIndex(1)
			else:
				self.FinalCoolMonth_Combo_NC.setCurrentIndex(self.InitialCoolMonth_Combo_NC.currentIndex() + 1)

		MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['InitialCoolMonth'] = self.InitialCoolMonth_Combo_NC.currentIndex() + 1
		MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['FinalCoolMonth'] = self.InitialCoolMonth_Combo_NC.currentIndex() + 1

	def ChangeTempDifference_NC(self):
		MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['delta_RAOA'] = float(self.TempDifference_Combo_NC.currentText().split()[0])

	def ChangeTempTolerance_NC(self):
		MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['margin_error'] = float(self.TempTolerance_Combo_NC.currentText().split()[0])

	def ChangeTimeBoundary_NC(self):
		if MainWindow.Result_NC[MainWindow.AHUidx_NC]['msg'].split('-')[0] != 'Success ':
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText(MainWindow.Result_NC[MainWindow.AHUidx_NC]['msg'].split('-')[1])
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				print('============ Popup Closed! ============')				
		MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['end_hour'] = int(self.FinalTime_Combo_NC.currentText().split(':')[0])
	
	def radioButtonClickedSeason_AOC(self):		
		self.InitialMonth_Combo_AOC.setCurrentIndex(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['InitialCoolMonth'] - 1)
		self.FinalMonth_Combo_AOC.setCurrentIndex(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['FinalCoolMonth'] - 1)
		self.OpeningTime_Combo_AOC.setCurrentIndex(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['OpeningTime'])
		self.ClosingTime_Combo_AOC.setCurrentIndex(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['ClosingTime'])
		self.TargetTempAHU_Combo_AOC.setCurrentIndex(MainWindow.Temp_tag_AOC.index(MainWindow.InputParam_AOC['AHU_TargetTemp']))
		self.ToleranceTemp_Combo_AOC.setCurrentIndex(MainWindow.TolTemp_tag_AOC.index(MainWindow.InputParam_AOC['AHU_TempTolerance']))		
	

	def ChangeAHUNum_AOC(self):
		MainWindow.AHUidx_AOC = int(self.AHUNum_Combo_AOC.currentIndex())
		self.AHUNum_Combo_AOC.setCurrentIndex(MainWindow.AHUidx_AOC)
		self.TargetTempAHU_Combo_AOC.setCurrentIndex(MainWindow.Temp_tag_AOC.index(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['AHU_TargetTemp']))
		self.ToleranceTemp_Combo_AOC.setCurrentIndex(MainWindow.TolTemp_tag_AOC.index(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['AHU_TempTolerance']))	
		## Load and show the subparameters of the selected AHU model	
		MainWindow.UpdateFigure(MainWindow, 'AOC')
			
	def ChangeMonthBoundary_AOC(self):
		if float(self.FinalMonth_Combo_AOC.currentIndex()) <= float(self.InitialMonth_Combo_AOC.currentIndex()):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText("Month boundary is wrong")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			msgBox.exec()
			if self.InitialMonth_Combo_AOC.currentIndex() == len(self.Month_tag_AOC) - 1:	# to check if InitialCoolMonth_Combo_CC is set to last index
				self.InitialMonth_Combo_AOC.setCurrentIndex(len(self.Month_tag_AOC) - 2)
				self.FinalMonth_Combo_AOC.setCurrentIndex(len(self.Month_tag_AOC) - 1)
			elif self.FinalMonth_Combo_AOC.currentIndex() == 0:	# to check if FinalCoolMonth_Combo_CC is set to first index
				self.InitialMonth_Combo_AOC.setCurrentIndex(0)
				self.FinalMonth_Combo_AOC.setCurrentIndex(1)
			else:
				self.FinalMonth_Combo_AOC.setCurrentIndex(self.InitialMonth_Combo_AOC.currentIndex() + 1)
		MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['InitialCoolMonth'] = self.InitialMonth_Combo_AOC.currentIndex() + 1
		MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['FinalCoolMonth'] = self.FinalMonth_Combo_AOC.currentIndex() + 1
	
	def radioButtonClickedMode_AOC(self):
		MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['SSRadioBtn'] = "Start" if MainWindow.radioStart_AOC.isChecked() else "Stop"
		MainWindow.UpdateFigure(MainWindow, 'AOC')
			
	def ChangeTimeBoundary_AOC(self):
		if float(self.ClosingTime_Combo_AOC.currentIndex()) <= float(self.OpeningTime_Combo_AOC.currentIndex()):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText("Time boundary is wrong (Initial month < Final month)")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			msgBox.exec()
			if self.OpeningTime_Combo_AOC.currentIndex() == len(MainWindow.Time_tag_AOC) - 1:	# to check if InitialTime_Combo_CC is set to last index
				self.OpeningTime_Combo_AOC.setCurrentIndex(len(MainWindow.Time_tag_AOC) - 2)
				self.ClosingTime_Combo_AOC.setCurrentIndex(len(MainWindow.Time_tag_AOC) - 1)
			elif self.ClosingTime_Combo_AOC.currentIndex() == 0:	# to check if FinalTime_Combo_CAOC is set to first index
				self.OpeningTime_Combo_AOC.setCurrentIndex(0)
				self.ClosingTime_Combo_AOC.setCurrentIndex(1)
			else:
				self.ClosingTime_Combo_AOC.setCurrentIndex(self.OpeningTime_Combo_AOC.currentIndex() + 1)
		MainWindow.InputParam_AOC['OpeningTime'] = int(self.OpeningTime_Combo_AOC.currentText().split(':')[0])	
		MainWindow.InputParam_AOC['ClosingTime'] = int(self.ClosingTime_Combo_AOC.currentText().split(':')[0])
		
	def ChangeAHUTargetTemp_AOC(self):
		MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['AHU_TargetTemp'] = float(self.TargetTempAHU_Combo_AOC.currentText().split()[0])
		
	def ChangeToleranceTime_AOC(self):
		MainWindow.InputParam_AOC['AHU_TempTolerance'] = float(self.ToleranceTemp_Combo_AOC.currentText().split()[0])
		
	def UpdateSummary(self, tab):
		if tab == 'CC':
			if MainWindow.Result_CC[MainWindow.AHUidx_CC]['msg'].split('-')[0] == 'Success ':
				tmp_text=[]
				for i_time in range(12, len(MainWindow.Result_CC[MainWindow.AHUidx_CC]['AHUStatus_optimul'])):
					if MainWindow.Result_CC[MainWindow.AHUidx_CC]['AHUStatus_optimul'][i_time] == 0:
						tmp_text.append('Off')
					else:
						tmp_text.append('On')
				MainWindow.AHUNum_SM_CC[MainWindow.AHUidx_CC].setText("[" + MainWindow.AHU_tag[MainWindow.AHUidx_CC] + "] " + str(tmp_text))
			else:
				MainWindow.AHUNum_SM_CC[MainWindow.AHUidx_CC].setText("[" + MainWindow.AHU_tag[MainWindow.AHUidx_CC] + "] " \
																		+ MainWindow.Result_CC[MainWindow.AHUidx_CC]['msg'].split('-')[1])
			
		if tab == 'EC' and MainWindow.Result_EC[MainWindow.AHUidx_EC]['msg'].split('-')[0] == 'Success ':
			MainWindow.AHUNum_SM_EC[MainWindow.AHUidx_EC].setText("[" + MainWindow.AHU_tag[MainWindow.AHUidx_EC] + "] " +
						str((MainWindow.Result_EC[MainWindow.AHUidx_EC]['ahu_damper_control'].flatten()[12:])) + " %"
						+ " Energy Savings : " + str(round(np.mean((MainWindow.Result_EC[MainWindow.AHUidx_EC]['y_prediction_c_1_Enthalpy'].flatten()))-np.mean((MainWindow.Result_EC[MainWindow.AHUidx_EC]['y_prediction_Enthalpy'].flatten())), 2)) + " Wh/kg")

		if tab == 'NC' and MainWindow.Result_NC[MainWindow.AHUidx_NC]['msg'].split('-')[0] == 'Success ':
			print(MainWindow.Result_NC[MainWindow.AHUidx_NC])
			MainWindow.AHUNum_SM_NC[MainWindow.AHUidx_NC].setText("[" + MainWindow.AHU_tag[MainWindow.AHUidx_NC] + "] " +
																  str(MainWindow.Result_NC[MainWindow.AHUidx_NC]['damp_on_best'][-4:]) + " %" +
																  ", Estimated Energy Savings : " + MainWindow.Result_NC[MainWindow.AHUidx_NC]['load_saving'] + " [kWh]")

		elif tab == 'NC' and MainWindow.Result_NC[MainWindow.AHUidx_NC]['msg'].split('-')[0] != 'Success ':
			MainWindow.AHUNum_SM_NC[MainWindow.AHUidx_NC].setText(
				"[" + MainWindow.AHU_tag[MainWindow.AHUidx_NC] + "] " + MainWindow.Result_NC[MainWindow.AHUidx_NC]['msg'].split('-')[1])

		if tab == 'AOC':
			if MainWindow.radioStart_AOC.isChecked():
				if MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Start']['success']:
					## set recommend time from status
					MainWindow.AHUNum_SM_AOC[MainWindow.AHUidx_AOC].setText("[" + MainWindow.AHU_tag[MainWindow.AHUidx_AOC] + "] " \
																		+ "Mode: Start, Start Time:" + str(MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Start']['time']))
				else:
					MainWindow.AHUNum_SM_AOC[MainWindow.AHUidx_AOC].setText("[" + MainWindow.AHU_tag[MainWindow.AHUidx_AOC] + "] " + "제어 가능 스케쥴이 없습니다.")
					
			else:
				if MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['success']:			
					## set recommend time from status
					MainWindow.AHUNum_SM_AOC[MainWindow.AHUidx_AOC].setText("[" + MainWindow.AHU_tag[MainWindow.AHUidx_AOC]  + "] " 
																		+ "Mode: Stop, Stop Time:" + str(MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['time']))
				else:
					MainWindow.AHUNum_SM_AOC[MainWindow.AHUidx_AOC].setText("[" + MainWindow.AHU_tag[MainWindow.AHUidx_AOC] + "] " + "제어 가능 스케쥴이 없습니다.")
	
	def UpdateExcel(self, tab):
		if tab == 'NC':
			print('update excel')
			
			wb = openpyxl.load_workbook(filename='D:\IPTower\설비 기동 상태 추천 (실증용).xlsx')
			#불러온 엑셀 파일 중 데이터를 찾을 sheet의 이름을 입력합니다.
			sheet1 = wb['Sheet']
			sheet1['B2'] = 'hello world!'
			wb.save("D:\IPTower\설비 기동 상태 추천 (실증용).xlsx")
			wb.close()



	def UpdateFigure(self, tab):			
		#### Cycle Control Tab ------------------------------------------			
		if tab == 'CC':
			MainWindow.fig_CC.clear()
			if MainWindow.Result_CC[MainWindow.AHUidx_CC]['msg'].split('-')[0] == 'Success ':
				## extend one step for status data
				AHUStatus_measured = MainWindow.Result_CC[MainWindow.AHUidx_CC]['AHUStatus_measured'].tolist() + [MainWindow.Result_CC[MainWindow.AHUidx_CC]['AHUStatus_measured'][-1]]
				RA_measured = MainWindow.Result_CC[MainWindow.AHUidx_CC]['RA_measured']
				## status should be shown with fill_between
				AHUStatus_controlled = MainWindow.Result_CC[MainWindow.AHUidx_CC]['AHUStatus_controlled'] + [MainWindow.Result_CC[MainWindow.AHUidx_CC]['AHUStatus_controlled'][-1]]
				AHUStatus_optimul = MainWindow.Result_CC[MainWindow.AHUidx_CC]['AHUStatus_optimul'] + [MainWindow.Result_CC[MainWindow.AHUidx_CC]['AHUStatus_optimul'][-1]]
				
				bottom_prev = []
				bottom_post = []
				for i in range(len(AHUStatus_measured)):
					bottom_prev.append(0)
				for i in range(len(AHUStatus_controlled)):
					bottom_post.append(0)
				x_prev = [i for i in range(len(bottom_prev))]
				xlabel_time = dt(MainWindow.currentDateTime.year, MainWindow.currentDateTime.month, MainWindow.currentDateTime.day, MainWindow.currentDateTime.hour, int(int(MainWindow.currentDateTime.minute/15)*15),0)
				x_prev_label_tmp = []
				for i in range(len(bottom_prev)):			
					x_prev_label_tmp.append((xlabel_time+timedelta(minutes=-(i+1)*15)).strftime("%H:%M"))
				x_prev_label_tmp.reverse()
				x_prev_tick = [0]
				x_prev_label = [x_prev_label_tmp[0]]
				for i in range(1,len(bottom_prev)):				
					if AHUStatus_measured[i-1] != AHUStatus_measured[i]:
						x_prev_tick.append(i)
						x_prev_label.append(x_prev_label_tmp[i])
						
				x_post = [i for i in range(len(bottom_post))]
				x_post_tick_controlled = [len(bottom_prev)-1]
				x_post_label_controlled = [xlabel_time.strftime("%H:%M")]
				x_post_tick_optimul = [len(bottom_prev)-1]
				x_post_label_optimul = [xlabel_time.strftime("%H:%M")]
				for i in range(1,len(bottom_post)):
					if AHUStatus_controlled[i-1] != AHUStatus_controlled[i] or i == len(bottom_post)-1:
						x_post_tick_controlled.append(len(bottom_prev)-1 + i)
						x_post_label_controlled.append((xlabel_time+timedelta(minutes=i*15)).strftime("%H:%M"))
					if AHUStatus_optimul[i-1] != AHUStatus_optimul[i] or i == len(bottom_post)-1:
						x_post_tick_optimul.append(len(bottom_prev)-1 + i)
						x_post_label_optimul.append((xlabel_time+timedelta(minutes=i*15)).strftime("%H:%M"))
				
				gs = GridSpec(nrows=4, ncols=1, width_ratios=[1], height_ratios=[1, 1, 1, 3])
				ax = MainWindow.fig_CC.add_subplot(gs[0, 0])
				ax.grid()
				ax.fill_between(x_prev, bottom_prev, AHUStatus_measured, step='post', alpha=0.35, hatch="//", edgecolor='white', facecolor='seagreen', label='Measured status')
				ax.set_xticks(x_prev_tick)
				ax.set_xticklabels(x_prev_label)
				ax.set_xlim([0, x_post[-1]])
				ax.set_yticks([0, 1])
				ax.set_yticklabels(['Off', 'On'])
				ax.set_ylim([0, 1])
				ax.set_ylabel('AHU status')
				ax.legend(loc='upper left')	
				ax.set_title('[' + MainWindow.Result_CC[MainWindow.AHUidx_CC]['PredictedDay'] + '] Cycle Control of '\
								+ str(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['AHU'][0:6]) + str(MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['AHU'][8]) + 'F)')
				
				ax2 = MainWindow.fig_CC.add_subplot(gs[1, 0])
				ax2.grid()
				#ax2.plot(x_prev, AHUStatus_measured, alpha=0.35, color='green')
				#ax2.plot(x_post, AHUStatus_controlled, alpha=0.35, color='red')
				ax2.fill_between(x_post, bottom_post, AHUStatus_controlled, step='post', alpha=0.35, hatch="//", edgecolor='white',  facecolor='red', label = 'Controlled status')	
				ax2.set_xticks(x_prev_tick + x_post_tick_controlled)
				ax2.set_xticklabels(x_prev_label + x_post_label_controlled)	
				ax2.set_xlim([0, x_post[-1]])	
				ax2.set_yticks([0, 1])
				ax2.set_yticklabels(['Off', 'On'])
				ax2.set_ylim([0, 1])
				ax2.set_ylabel('AHU status')
				ax2.legend(loc='upper left')	
				
				ax3 = MainWindow.fig_CC.add_subplot(gs[2, 0])
				ax3.grid()
				ax3.fill_between(x_post, bottom_post, AHUStatus_optimul, step='post', alpha=0.35, hatch="//", edgecolor='white', facecolor='royalblue', label='Optimul status (Recommended)')
				ax3.set_xticks(x_prev_tick + x_post_tick_optimul)
				ax3.set_xticklabels(x_prev_label + x_post_label_optimul)	
				ax3.set_xlim([0, x_post[-1]])				
				ax3.set_yticks([0, 1])
				ax3.set_yticklabels(['Off', 'On'])
				ax3.set_ylim([0, 1])
				ax3.set_ylabel('AHU status')
				ax3.legend(loc='upper left')	
						
				ax4 = MainWindow.fig_CC.add_subplot(gs[3, 0])
				ax4.grid()
				ax4.plot(x_prev[:-1], RA_measured, 'o-', alpha=0.35, color = 'green', label='Measured value')
				ax4.plot(x_post[:-1], MainWindow.Result_CC[MainWindow.AHUidx_CC]['RA_controlled'], 's--', alpha=0.35, color='red', label = 'Predicted value (Controlled)')
				ax4.plot(x_post[:-1], MainWindow.Result_CC[MainWindow.AHUidx_CC]['RA_optimul'], 'X--', alpha=0.35, color='blue', label = 'Predicted value (Optimul)')
				ax4.axhline(y=MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['LowerTemp'], color='gray', linestyle='dotted', label='Setting temp. (Lower limit)')
				ax4.axhline(y=MainWindow.InputParam_CC[MainWindow.AHUidx_CC]['UpperTemp'], color='gray', linestyle='-.', label='Setting temp. (Upper limit')
				ax4.set_xticks(x_prev_tick + x_post_tick_optimul)
				ax4.set_xticklabels(x_prev_label + x_post_label_optimul)
				ax4.set_xlabel('Time', fontsize=10)
				ax4.set_xlim([0, x_post[-1]])
				ax4.set_ylim([MainWindow.Temp_tag_CC[0], MainWindow.Temp_tag_CC[-1]])
				ax4.set_ylabel('RA Temperature [℃]')
				ax4.legend(loc = 'upper left')				
						
			MainWindow.fig_CC.tight_layout()
			MainWindow.fig_CC.subplots_adjust(wspace=0, hspace=0.5)
			MainWindow.canvas_CC.draw()		
			
		#### Enthalpy Control Tab ------------------------------------------	
		if tab == 'EC' and MainWindow.Result_EC[MainWindow.AHUidx_EC]['msg'].split('-')[0] == 'Success ':
			MainWindow.fig_EC.clear()
			x_CalAmount_prev = [i for i in range(len(MainWindow.Result_EC[MainWindow.AHUidx_EC]['ahu_damper_BEMS']))]

			damper_BEMS = MainWindow.Result_EC[MainWindow.AHUidx_EC]['ahu_damper_BEMS'].flatten()
			damper_BEMS = np.append(damper_BEMS[:12], damper_BEMS[11])

			damp_pred = MainWindow.Result_EC[MainWindow.AHUidx_EC]['ahu_damper_pred'].flatten()
			damp_pred = np.append(damp_pred, damp_pred[3])

			damper_control = MainWindow.Result_EC[MainWindow.AHUidx_EC]['ahu_damper_control'].flatten()
			damper_control = damper_control[12:]
			damp_control = np.append(damper_control, damper_control[3])

			control_label = MainWindow.Result_EC[MainWindow.AHUidx_EC]['label'][12:]

			gs = GridSpec(nrows=4, ncols=1, width_ratios=[1], height_ratios=[1, 1, 1.5, 1.5])
			#ax = MainWindow.fig_NC.add_subplot(2, 1, 1)
			ax = MainWindow.fig_EC.add_subplot(gs[0, 0])
			ax.grid()

			## damper 개도율 plot 변경
			a = 0
			for i in range(len(damper_BEMS) - 1):
				if damper_BEMS[i] != damper_BEMS[i + 1]:
					a = i

			if a!= 0:
				damper1 = np.append(damper_BEMS[:a + 1], damper_BEMS[a])
				damper2 = damper_BEMS[a + 1:]

				#ax.plot(np.arange(0, a+2), damper1, label='Measured Value', color='teal', alpha=0.35)
				ax.fill_between(np.arange(0, a+2), damper1, label='Measured Value', alpha=0.35, step='post', hatch="//", edgecolor='white', color='teal')
				#ax.plot(np.arange(a+1, 13), damper2, color='teal', alpha=0.35)
				ax.fill_between(np.arange(a+1, 13), damper2, alpha=0.35, step='post', hatch="//", edgecolor='white', color='teal')
			else:
				#ax.plot(np.arange(0, 13), damper_BEMS[:13], label='Measured Value', color='teal', alpha=0.35)
				ax.fill_between(np.arange(0, 13), damper_BEMS[:13], label='Measured Value', alpha=0.35, step='post', hatch="//", edgecolor='white', color='teal')

			#ax.plot(np.arange(12, 17), damp_pred, label='Predicted Value', color='#FF445F', alpha=0.35)
			ax.fill_between(np.arange(12, 17), damp_pred, label='Predicted Value', color='#FF445F', alpha=0.35, step='post', hatch="//", edgecolor='white')
			ax.axes.xaxis.set_visible(False)
			ax.set_xlim(0, 16)
			ax.set_xticks([i for i in range(0,len(MainWindow.Result_EC[MainWindow.AHUidx_EC]['label']),4)])
			# ax.set_xticklabels(MainWindow.Result_EC['label'])
			ax.set_ylim(0,100)
			ax.set_ylabel('Damper open ratio [%]')
			# ax.set_title('Enthalpy Control of AHU-' + str(MainWindow.AHUnum_EC))
			ax.set_title('[' + MainWindow.Result_EC[MainWindow.AHUidx_EC]['PredictedDay'] + '] Enthalpy Control of '\
								+ str(MainWindow.InputParam_EC[MainWindow.AHUidx_EC]['AHU'][0:6]) + str(MainWindow.InputParam_EC[MainWindow.AHUidx_EC]['AHU'][8]) + 'F)')
			ax.legend(loc='upper left')
			
			ax2 = MainWindow.fig_EC.add_subplot(gs[1, 0])
			ax2.grid()

			#ax2.plot(np.arange(12, 17), damp_control, label='Controlled Value', color='#FEB630', alpha=0.55)
			ax2.fill_between(np.arange(12, 17), damp_control, label='Controlled Value', alpha=0.55, color='#FEB630', step='post', hatch="//", edgecolor='white')
			ax2.axes.xaxis.set_visible(False)
			ax2.set_xlim(0, 16)
			ax2.set_xticks([12])
			ax2.set_xticklabels(control_label[0::4])
			ax2.set_ylim(0,100)
			ax2.set_ylabel('Damper open ratio [%]')
			# ax.set_title('Enthalpy Control of AHU-' + str(MainWindow.AHUnum_EC))
			ax2.legend(loc='upper left')

			measured_MA = MainWindow.Result_EC[MainWindow.AHUidx_EC]['ahu_damper_BEMS']
			predicted_MA = MainWindow.Result_EC[MainWindow.AHUidx_EC]['y_prediction_ma']
			# predicted_MA = np.append(measured_MA[-1], predicted_MA)
			controlled_MA = MainWindow.Result_EC[MainWindow.AHUidx_EC]['y_prediction_ma_c'].flatten()
			# controlled_MA = np.append(measured_MA[-1], controlled_MA)

			ax3 = MainWindow.fig_EC.add_subplot(gs[2, 0])
			ax3.grid()
			ax3.set_xlim(0, 16)
			ax3.set_xticks([i for i in range(0,len(MainWindow.Result_EC[MainWindow.AHUidx_EC]['label']),4)])
			# ax2.set_xticklabels(MainWindow.Result_EC['label'])
			ax3.set_ylabel('MA tempearture [℃]')
			# ax2.plot(np.arange(0, 16), measured_MA, label='Damper Open Ratio_BEMS', color='green')
			ax3.plot(np.arange(0, 12), controlled_MA[:12], 'D-', label='Measured Value', color='teal', alpha=0.35)
			# ax2.plot(np.arange(1, 13), MainWindow.Result_EC['real_p_ma'], 'D-', label='BEMS_predicted', color='#FF445F', alpha=0.55)
			ax3.plot(np.arange(12, 16), predicted_MA[12:], 'D--', label='Predicted Value', color='#FF445F', alpha=0.55)
			ax3.plot(np.arange(12, 16), controlled_MA[12:], 'D--', label='Controlled Value', color='#FEB630', alpha=0.55)
			ax3.set_ylim(10, 35)
			ax3.legend(loc='upper left')
			ax3.axes.xaxis.set_visible(False)
			# ax2.axvline(x=12, color='gray', linestyle='--', linewidth=3)

			measured_coil = MainWindow.Result_EC[MainWindow.AHUidx_EC]['Enthalpy_BEMS'].flatten()
			predicted_coil = MainWindow.Result_EC[MainWindow.AHUidx_EC]['y_prediction_Enthalpy'].flatten()
			# predicted_coil = np.append(measured_coil[-1], predicted_coil)
			controlled_coil = MainWindow.Result_EC[MainWindow.AHUidx_EC]['y_prediction_c_1_Enthalpy'].flatten()
			# controlled_coil = np.append(measured_coil[-1], controlled_coil)

			ax4 = MainWindow.fig_EC.add_subplot(gs[3, 0])
			ax4.grid()
			# ax3.set_xlim(0, 17)
			# ax3.set_xticklabels(MainWindow.Result_EC['label'])
			ax4.set_xlim(0, 16)
			ax4.set_xticks([i for i in range(0,len(MainWindow.Result_EC[MainWindow.AHUidx_EC]['label']),4)])
			ax4.set_xticklabels(MainWindow.Result_EC[MainWindow.AHUidx_EC]['label'][0::4])
			ax4.set_xlim(0,16,2)

			ax4.plot(np.arange(0, 12), measured_coil[:12], 'D-', label='Enthalpy_BEMS', color='teal', alpha=0.35)
			ax4.plot(np.arange(12, 16), predicted_coil[12:], 'D-', label='Predicted Value', color='#FF445F', alpha=0.55)
			ax4.plot(np.arange(12, 16), controlled_coil[12:], 'D--', label='Controlled Value', color='#FEB630', alpha=0.55)

			# ax3.bar(np.arange(1, 13), measured_coil, width=0.2, edgecolor="white", label='Enthalpy_BEMS', color='teal', alpha=0.35)
			# val1 = ax3.bar(np.arange(13, 17)-0.15, predicted_coil, width=0.2, edgecolor="white", label='Predicted Value', color='#FF445F', alpha=0.55)
			# val2 = ax3.bar(np.arange(13, 17)+0.15, controlled_coil, width=0.2, edgecolor="white", label='Controlled Value', color='#FEB630', alpha=0.55)

			def autolabel(rects):
				for rect in rects:
					h = rect.get_height()
					# ax3.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')

			# autolabel(val1)
			# autolabel(val2)

			# ax3.bar(np.arange(12, 16), predicted_coil, width = 0.3, edgecolor="black", label='Predicted Value', color='red')
			# ax3.bar(np.arange(12, 16), controlled_coil, width = 0.3, edgecolor="black", label='Controlled Value', color='gold')
			# ax3.p	lot(np.arange(0, 16), controlled_MA, 'D--', label='MA Temperature (Controlled)', color='violet')
			ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
			# ax3.axvline(x=12, color='gray', linestyle='--', linewidth=3)

			# ax3.plot(MainWindow.Result_EC['y_prediction_coil'],  label='Predicted Value', color='mediumseagreen')
			# ax3.plot(MainWindow.Result_EC['y_prediction_c_1_coil'], '<--', label='Controlled Value', color='khaki')

			ax4.set_ylabel('MA Enthalpy [Wh/kg]')
			ax4.legend(loc='best')

			MainWindow.fig_EC.tight_layout()
			MainWindow.fig_EC.subplots_adjust(wspace=0, hspace=0.25)
			
			MainWindow.canvas_EC.draw()
			MainWindow.Saving_tab_EC.setText("*** Estimated Energy savings : " + str(round(np.mean((MainWindow.Result_EC[MainWindow.AHUidx_EC]['y_prediction_c_1_Enthalpy'].flatten()))-np.mean((MainWindow.Result_EC[MainWindow.AHUidx_EC]['y_prediction_Enthalpy'].flatten())), 2)) + " Wh/Kg")

		#### Night Purge Control Tab ------------------------------------------
		if tab == 'NC':
			MainWindow.fig_NC.clear()
			MainWindow.NC_able_tab.setText("나이트퍼지 구동 가능시간 : 18:00 이후")
			MainWindow.NC_saving_tab.setText(" ")
			if MainWindow.Result_NC[MainWindow.AHUidx_NC]['msg'].split('-')[0] == 'Success ':
			
				# plt.rcParams['font.family'] = 'Times New Roman'
				# plt.rcParams['font.size'] = 25
				
				on_idx_ = MainWindow.Result_NC[MainWindow.AHUidx_NC]['on_idx'][0]
				
				x_Damper = []
				DamperBottom = []
				DamperStatus = []
				AvailableArea = []
				for i in range(len(MainWindow.Result_NC[MainWindow.AHUidx_NC]['damp_on_best'])):
					x_Damper.append(i)
					DamperBottom.append(0)
					DamperStatus.append(MainWindow.Result_NC[MainWindow.AHUidx_NC]['damp_on_best'][i])
					if i >= on_idx_:
						AvailableArea.append(100)
					else:
						AvailableArea.append(0)
					
				x_Damper.append(x_Damper[-1])
				DamperBottom.append(DamperBottom[-1])
				DamperStatus.append(DamperStatus[-1])
				AvailableArea.append(AvailableArea[-1])
				
				gs = GridSpec(nrows=2, ncols=1, width_ratios=[1], height_ratios=[3, 1])
				#ax = MainWindow.fig_NC.add_subplot(2, 1, 1)
				ax = MainWindow.fig_NC.add_subplot(gs[0, 0])
				ax.grid()

				ax.fill_between(x_Damper, DamperBottom, AvailableArea, alpha=0.15, step='post', hatch="//", edgecolor='white', color='lightskyblue', label='Controllable time',)
				ax.plot(MainWindow.Result_NC[MainWindow.AHUidx_NC]['RA_asis'], color='k', ls = '--', label = '$T_{RA}$  (As-Is)')
				ax.plot(MainWindow.Result_NC[MainWindow.AHUidx_NC]['RA_tobe'], color='r', label='$T_{RA}$ (To-Be)', marker = 'o', markersize = 10, mfc='none')
				ax.plot(MainWindow.Result_NC[MainWindow.AHUidx_NC]['RA_fail'], color='#FEB630', ls='--', label='RA_fail')
				[ax.plot(MainWindow.Result_NC[MainWindow.AHUidx_NC]['RA_proposal'][i], ls='--', lw=1, color='#FEB630') \
							for i in range(1, len(MainWindow.Result_NC[MainWindow.AHUidx_NC]['RA_proposal']) - 1)]
				ax.plot(MainWindow.Result_NC[MainWindow.AHUidx_NC]['OA'], color='y', label = '$T_{OA}$')
				ax.set_xticks(np.arange(0, MainWindow.Result_NC[MainWindow.AHUidx_NC]['future_num'], 4))
				ax.set_xticklabels([])
				ax.set_xlim([0, MainWindow.Result_NC[MainWindow.AHUidx_NC]['future_num']-1])
				ax.set_ylim([min(MainWindow.Result_NC[MainWindow.AHUidx_NC]['OA'])-3, max(MainWindow.Result_NC[MainWindow.AHUidx_NC]['RA_asis'])+3])
				ax.legend()
				ax.set_ylabel('Temperature ($^\circ$C)')
				# ax.set_title(MainWindow.Result_NC[MainWindow.AHUidx_NC]['title'])
				ax.set_title('[' + MainWindow.Result_NC[MainWindow.AHUidx_NC]['title'] + '] Night Purge Control of '\
								+ str(MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['AHU'][0:6]) + str(MainWindow.InputParam_NC[MainWindow.AHUidx_NC]['AHU'][8]) + 'F)')
				ax.legend(loc='best')
				
				
				#ax1 = MainWindow.fig_NC.add_subplot(2, 1, 2)
				ax1 = MainWindow.fig_NC.add_subplot(gs[1, 0])
				ax1.grid()
				#ax1.plot(MainWindow.Result_NC[MainWindow.AHUidx_NC]['damp_on_best'], color='teal', label='To-Be', alpha=0.35)
				#ax1.fill_between(np.arange(0, 40), MainWindow.Result_NC[MainWindow.AHUidx_NC]['damp_on_best'], alpha=0.35, color='teal')
				ax1.fill_between(x_Damper, DamperBottom, DamperStatus, alpha=0.35, step='post', hatch="//", edgecolor='white', color='teal', label='To-Be',)
				'''여기까지요!'''
				
				#ax1.fill_between(x_, bottom_damp_status, AHUStatus_measured, step='post', alpha=0.35, facecolor='green', label='Measured status')
				ax1.set_xticks(np.arange(0, MainWindow.Result_NC[MainWindow.AHUidx_NC]['future_num'], 4))
				ax1.set_xticklabels(MainWindow.Result_NC[MainWindow.AHUidx_NC]['x_label'])
				ax1.set_xlim([0, MainWindow.Result_NC[MainWindow.AHUidx_NC]['future_num']-1])
				ax1.set_ylabel('Damper open rate (%)') # chk
				ax1.set_ylim(MainWindow.Result_NC[MainWindow.AHUidx_NC]['damper_ratio']) # chk			
				ax1.set_yticks(MainWindow.Result_NC[MainWindow.AHUidx_NC]['damper_ratio'])
				ax1.legend()
				ax1.set_yticklabels(['0', '100'])

				MainWindow.NC_saving_tab.setText("*** Estimated Energy savings : " + MainWindow.Result_NC[MainWindow.AHUidx_NC]['load_saving'] + " kWh")

			MainWindow.fig_NC.tight_layout()
			MainWindow.fig_NC.subplots_adjust(wspace=0, hspace=0.15)
			MainWindow.canvas_NC.draw()
		
		#### AHU Optimul Control Tab ------------------------------------------			
		if tab == 'AOC':
			MainWindow.fig_AOC.clear()
			## Start mode
			if MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['SSRadioBtn'] == "Start":	
				if MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Start']['success']:
					#### AHU -----------------------------------------
					# For plot status 						
					AHUStatus_rec = []
					AHUBottom_rec = []
					for i in range(len(MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Start']['table']['AHU OP'].tolist())-8):
						AHUStatus_rec.append(MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Start']['table']['AHU OP'].tolist()[i+8])
						AHUBottom_rec.append(0)
					
					# For plot status 
					AHUStatus_rec.append(AHUStatus_rec[-1])
					AHUBottom_rec.append(AHUBottom_rec[-1])
					
					x_AHUTemp = [i for i in range(len(AHUStatus_rec)-1)]
					x_AHUStatus = [i for i in range(len(AHUStatus_rec))]
					
					x_tick_ahu = []
					x_ticklabel_ahu = []
					for i in range(len(AHUBottom_rec)):
						if i%4 == 0 or AHUStatus_rec[i-1] != AHUStatus_rec[i]:
							x_tick_ahu.append(i)
							x_ticklabel_ahu.append((xlabel_time+timedelta(minutes=i*15)).strftime("%H:%M"))						

					gs = GridSpec(nrows=3, ncols=1, width_ratios=[1], height_ratios=[1, 3, 1])
					#ax3 = MainWindow.fig_CAOC.add_subplot(4, 1, 1)
					ax = MainWindow.fig_AOC.add_subplot(gs[0, 0])
					ax.grid()
					ax.fill_between(x_CoolerStatus, CoolerBottom_rec, CoolerStatus_rec, step='post', facecolor='royalblue', hatch="//", alpha=0.35, edgecolor='white', label='Coil status')
					ax.set_xticks(x_tick_cooler)
					ax.set_xticklabels(x_ticklabel_cooler)
					ax.set_xlim([0, 16])
					ax.set_yticks([0, 1])
					ax.set_yticklabels(['Off', 'On'])
					ax.set_ylim([0, 1])
					#ax.set_ylabel('Cooler status')
					ax.legend(loc='upper right')	
					predDate = str(MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['PredictedDate'])
					ax.set_title('[' + predDate.split(',')[0] + ' -' + predDate.split(',')[1] + ' -' + predDate.split(',')[2] + '] AHU Optimul Control -' +\
										str(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['AHU'][0:6]) + str(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['AHU'][8]) + 'F)')	
					
					#ax4 = MainWindow.fig_CAOC.add_subplot(4, 1, 3)
					ax4 = MainWindow.fig_AOC.add_subplot(gs[1, 0])
					ax4.grid()
					for i in range(len(MainWindow.Result_CAOC['AHU_Start'][MainWindow.AHUidx_CAOC]['df'].columns)):
						if i == 0:
							ax4.plot(x_AHUTemp, MainWindow.Result_CAOC['AHU_Start'][MainWindow.AHUidx_CAOC]['df'].iloc[1:,i].values, '--', color='black', alpha=0.35, label = 'Failure')
						else:
							ax4.plot(x_AHUTemp, MainWindow.Result_CAOC['AHU_Start'][MainWindow.AHUidx_CAOC]['df'].iloc[1:,i].values, '--', color='black', alpha=0.35)
					#ax4.plot(x_ra_temp_measured, MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['temp'], label = 'Measured') 	## Previous mesured RA
					ax4.plot(x_AHUTemp, MainWindow.Result_CAOC['AHU_Start'][MainWindow.AHUidx_CAOC]['table']['RA'][8:], marker = 'o', markersize = 10, mfc='none', color='seagreen', label = 'Success')
					ax4.axhline(y=MainWindow.Result_CAOC['AHU_Start'][MainWindow.AHUidx_CAOC]['target temp'], color='red', alpha=0.5, label='Set temperature & Target time')
					ax4.axvline(x=len(x_AHUTemp)-4, color='red', alpha=0.5)
					ax4.set_xlim([0, 16])
					ax4.set_ylabel('RA Temperature [℃]')
					ax4.axes.xaxis.set_visible(True)
					ax4.legend(loc='upper right')	
					ylim_max = ax4.get_ylim()[-1]
					ylim_min = ax4.get_ylim()[0]
					ax4.set_ylim([ylim_min - 1, ylim_max + 1])	
					ax4.axes.xaxis.set_visible(False)
					
					#ax2 = MainWindow.fig_CAOC.add_subplot(4, 1, 4)
					ax2 = MainWindow.fig_AOC.add_subplot(gs[2, 0])
					ax2.grid()
					ax2.fill_between(x_AHUStatus, AHUBottom_rec, AHUStatus_rec, step='post', facecolor='tomato', hatch="//", alpha=0.35, edgecolor='white', label='AHU status')
					#ax2.axvline(x=len(x_ra_temp_measured)-1, color='gray', linestyle='--', linewidth=3)
					ax2.set_xticks(x_tick_ahu)
					ax2.set_xticklabels(x_ticklabel_ahu)
					ax2.set_xlim([0, 16])
					ax2.set_yticks([0, 1])
					ax2.set_yticklabels(['Off', 'On'])
					ax2.set_ylim([0, 1])
					#ax2.set_ylabel('AHU Status')
					ax2.legend(loc='upper right')
					#ax2.set_title('AHU Optimul Schedule (AHU - ' + str(MainWindow.AHUnum_CHAOC) + ')')
				
							
			## Stop mode
			elif MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['SSRadioBtn'] == "Stop":
				if MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['success']:
				#### AHU -----------------------------------------
				## extend one step for status data	
				
					xlabel_time = dt.strptime(str(MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['table'].index[8]), '%Y-%m-%d %H:%M:%S')
					#### AHU -----------------------------------------
					# For plot status 						
					AHUStatus_rec = []
					AHUBottom_rec = []
					for i in range(len(MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['table']['AHU OP'].tolist())-8):
						AHUStatus_rec.append(MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['table']['AHU OP'].tolist()[i+8])
						AHUBottom_rec.append(0)
					
					# For plot status 
					AHUStatus_rec.append(AHUStatus_rec[-1])
					AHUBottom_rec.append(AHUBottom_rec[-1])	
					
					x_AHUTemp = [i for i in range(len(AHUStatus_rec)-1)]
					x_AHUStatus = [i for i in range(len(AHUStatus_rec))]
					
					x_tick_ahu = []
					x_ticklabel_ahu = []
					for i in range(len(AHUBottom_rec)):
						if i%4 == 0 or AHUStatus_rec[i-1] != AHUStatus_rec[i]:
							x_tick_ahu.append(i)
							x_ticklabel_ahu.append((xlabel_time+timedelta(minutes=i*15)).strftime("%H:%M"))

					gs = GridSpec(nrows=3, ncols=1, width_ratios=[1], height_ratios=[1, 3, 1])
					
					#ax = MainWindow.fig_CAOC.add_subplot(4, 1, 2)
					ax = MainWindow.fig_AOC.add_subplot(gs[0, 0])
					ax.grid()
					ax.fill_between(x_AHUStatus, AHUBottom_rec, AHUStatus_rec, step='post', facecolor='royalblue', hatch="//", alpha=0.35, edgecolor='white', label='Gas usage [%]')
					#ax2.axvline(x=len(x_ra_temp_measured)-1, color='gray', linestyle='--', linewidth=3)
					ax.set_xticks(x_tick_ahu)
					ax.set_xticklabels(x_ticklabel_ahu)
					ax.set_xlim([0, 16])
					ax.set_yticks([0, 1])
					ax.set_yticklabels(['0', '100'])
					ax.set_ylim([0, 1])
					ax.legend(loc='upper right')
					predDate = str(MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['PredictedDate'])
					ax.set_title('[' + predDate.split(',')[0] + ' -' + predDate.split(',')[1] + ' -' + predDate.split(',')[2] + '] AHU Optimul Control - ' +\
										str(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['AHU'][0:6]) + str(MainWindow.InputParam_AOC[MainWindow.AHUidx_AOC]['AHU'][8]) + 'F)')	
					
					
					#ax4 = MainWindow.fig_CAOC.add_subplot(4, 1, 3)
					ax4 = MainWindow.fig_AOC.add_subplot(gs[1, 0])
					ax4.grid()
					for i in range(len(MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['df'].columns)):
						if i == 0:
							ax4.plot(x_AHUTemp, MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['df'].iloc[1:,i].values, '--', color='black', alpha=0.35, label = 'Failure')
						else:
							ax4.plot(x_AHUTemp, MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['df'].iloc[1:,i].values, '--', color='black', alpha=0.35)
					#ax4.plot(x_ra_temp_measured, MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['temp'], label = 'Measured') 	## Previous mesured RA
					ax4.plot(x_AHUTemp, MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['table']['RA'][8:], marker = 'o', markersize = 10, mfc='none', color='seagreen', label = 'Success')
					ax4.axhline(y=MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['target temp'], color='red', alpha=0.5, label='Set temperature & Target time')
					ax4.axvline(x=len(x_AHUTemp)-4, color='red', alpha=0.5)
					ax4.set_xlim([0, 16])
					ax4.set_ylabel('RA Temperature [℃]')
					ax4.axes.xaxis.set_visible(True)
					ax4.legend(loc='upper right')	
					ylim_max = ax4.get_ylim()[-1]
					ylim_min = ax4.get_ylim()[0]
					ax4.set_ylim([ylim_min - 1, ylim_max + 1])
					ax4.axes.xaxis.set_visible(False)
					
					#ax2 = MainWindow.fig_CAOC.add_subplot(4, 1, 4)
					ax2 = MainWindow.fig_AOC.add_subplot(gs[2, 0])
					ax2.grid()
					ax2.fill_between(x_AHUStatus, AHUBottom_rec, AHUStatus_rec, step='post', facecolor='tomato', hatch="//", alpha=0.35, edgecolor='white', label='AHU status')
					#ax2.axvline(x=len(x_ra_temp_measured)-1, color='gray', linestyle='--', linewidth=3)
					ax2.set_xticks(x_tick_ahu)
					ax2.set_xticklabels(x_ticklabel_ahu)
					ax2.set_xlim([0, 16])
					ax2.set_yticks([0, 1])
					ax2.set_yticklabels(['Off', 'On'])
					ax2.set_ylim([0, 1])
					#ax2.set_ylabel('AHU Status')
					ax2.legend(loc='upper right')	
					#ax2.set_title('AHU Optimul Schedule (AHU - ' + str(MainWindow.AHUnum_CHAOC) + ')')
				
				
			elif MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Start']['msg'].split('-')[0] == 'Failure ' or MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['msg'].split('-')[0] == 'Failure ':
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Information)
				if MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Start']['msg'].split('-')[0] == 'Failure ':
					msgBox.setText("[AHU Optimul Control]" + MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Start']['msg'].split('-')[1])
				if MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['msg'].split('-')[0] == 'Failure ':
					msgBox.setText("[AHU Optimul Control]" + MainWindow.Result_AOC[MainWindow.AHUidx_AOC]['AHU_Stop']['msg'].split('-')[1])
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Ok)
				msgBox.exec()	
				
			MainWindow.fig_AOC.tight_layout()
			MainWindow.fig_AOC.subplots_adjust(wspace=0, hspace=0.25)
			MainWindow.canvas_AOC.draw()
		
		
	def ModuleExe(self, tab):
		
		#### Cycle Control Tab ------------------------------------------	
		if tab == 'CC':
			QApplication.setOverrideCursor(Qt.WaitCursor)
			MainWindow.Result_CC[MainWindow.AHUidx_CC] = CC.main_CC(MainWindow.currentDateTime, MainWindow.InputParam_CC, MainWindow.AHUidx_CC)
			QApplication.restoreOverrideCursor()
			MainWindow.UpdateSummary(MainWindow, 'CC')
		#### ------------------------------------------	
		#### Enthalpy Control Tab ------------------------------------------	
		elif tab == 'EC':
			QApplication.setOverrideCursor(Qt.WaitCursor)			
			MainWindow.Result_EC[MainWindow.AHUidx_EC] = EC.main_EC(MainWindow.currentDateTime, MainWindow.InputParam_EC, MainWindow.AHUidx_EC)
			QApplication.restoreOverrideCursor()
			MainWindow.UpdateSummary(MainWindow, 'EC')

		#### Night Purge Control Tab ------------------------------------------
		elif tab == 'NC':
			QApplication.setOverrideCursor(Qt.WaitCursor)
			MainWindow.Result_NC[MainWindow.AHUidx_NC] = NC.main_NC(MainWindow.currentDateTime, MainWindow.InputParam_NC, MainWindow.AHUidx_NC)
			QApplication.restoreOverrideCursor()
			MainWindow.UpdateSummary(MainWindow, 'NC')
			MainWindow.UpdateExcel(MainWindow, 'NC')
		#### AHU Optimul Control Tab ------------------------------------------			
		elif tab == 'AOC':
			QApplication.setOverrideCursor(Qt.WaitCursor)
			MainWindow.Result_AOC[MainWindow.AHUidx_AOC] = AOC.main_AOC(MainWindow.currentDateTime, MainWindow.InputParam_AOC, MainWindow.AHUidx_AOC)
			QApplication.restoreOverrideCursor()
			MainWindow.UpdateSummary(MainWindow, 'AOC')
		#### ------------------------------------------
			
	def ClickedUpdateData_CC(self):
		# Update current time		
		MainWindow.currentDateTime = dt.now()		
		# Update the model 
		tmpAHUidx = MainWindow.AHUidx_CC
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUidx_CC = i
			MainWindow.ModuleExe(MainWindow, 'CC')
			if i == tmpAHUidx:
				MainWindow.UpdateFigure(MainWindow, 'CC')
		MainWindow.AHUidx_CC = tmpAHUidx
		
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.FailureMsg(MainWindow, 'CC', i)		# Failure message			
		#### ------------------------------------------
		
	def ClickedUpdateData_EC(self):			
		## Update current time		
		MainWindow.currentDateTime = dt.now()
		#### Update every ?? ------------------------------------------
		##Update the model
		tmpAHUidx = MainWindow.AHUidx_EC
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUidx_EC = i
			MainWindow.ModuleExe(MainWindow, 'EC')
			if i == tmpAHUidx:
				MainWindow.UpdateFigure(MainWindow, 'EC')
		MainWindow.AHUidx_EC = tmpAHUidx
		#### ------------------------------------------

	def ClickedUpdateData_NC(self):
		## Update current time
		MainWindow.currentDateTime = dt.now()
		##Update the model
		tmpAHUidx = MainWindow.AHUidx_NC
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.AHUidx_NC = i
			MainWindow.ModuleExe(MainWindow, 'NC')
			if i == tmpAHUidx:
				MainWindow.UpdateFigure(MainWindow, 'NC')
		MainWindow.AHUidx_NC = tmpAHUidx

		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.FailureMsg(MainWindow, 'NC', i)		# Failure message
			
	def ClickedUpdateData_AOC(self):
		# Update current time		
		MainWindow.currentDateTime = dt.now()
		# Update the model 
		MainWindow.ModuleExe(MainWindow, 'CHAOC')
		MainWindow.UpdateFigure(MainWindow, 'CHAOC')
		
		for i in range(len(MainWindow.AHU_tag)):
			MainWindow.FailureMsg(MainWindow, 'CHAOC', i)		# Failure message
		
		
	def FailureMsg(self, tab, AHUidx):
		if tab == 'CC' and MainWindow.Result_CC[AHUidx]['msg'].split('-')[0] == 'Failure ':		
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText("[Cycle Control]" + MainWindow.Result_CC[AHUidx]['msg'].split('-')[1] + " (" + str(MainWindow.InputParam_CC[AHUidx]['AHU']) + ")")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			msgBox.exec()	
			
		if tab == 'NC' and MainWindow.Result_NC[AHUidx]['msg'].split('-')[0] != 'Success ':
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText("[Night Purge Control]" + str(MainWindow.Result_NC[AHUidx]['msg'].split('-')[1]))
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			msgBox.exec()
			
		if tab == 'CAOC2' and MainWindow.Result_CAOC['AHU_Start'][AHUidx]['msg'].split('-')[0] == 'Failure ':		
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText("[AHU Optimal Control]" + MainWindow.Result_CAOC['AHU_Start'][AHUidx]['msg'].split('-')[1] + " (AHU-" + str(MainWindow.InputParam_CAOC[AHUidx]['AHU']) + ")")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			msgBox.exec()	
			
	def closeEvent(self, event):
		reply = QMessageBox.question(self, 'Message', 'Are you sure to quit?',
									QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
		if reply == QMessageBox.Yes:
			event.accept()
		else:
			event.ignore()
			

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	app.exec_()
		
	################ 참고 #######################

	# setEnalbed(): False 설정 시, 버튼을 사용할 수 없습니다.
	# btn3.setEnabled(False)
	
	# ax 에 x 축을 설정해준다
	#ax.xaxis.set_major_locator(weeks)
	#ax.xaxis.set_major_formatter(weeksFmt)
	
	#weeks = mdates.WeekdayLocator(mdates.MONDAY) # x 축 어디에 찍을것인지 지정
	#weeksFmt = mdates.DateFormatter('%m.%d')     # 어떻게 표시할것인지 설정
	
	# self.setWindowIcon(QIcon('icon.png'))
	
	#self.lineEdit1 = QLineEdit()		# 텍스트 입력 상자