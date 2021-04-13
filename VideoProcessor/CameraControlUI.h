#pragma once

#include "ApplicationGraph.h"

void camera_control_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class CameraControlFrame : public wxFrame {
	int node_graph_id;
	int node_id;
public:
	wxTextCtrl* tc_camera_count;

	wxTextCtrl* tc_m_path;
	wxTextCtrl* tc_s_path;
	wxTextCtrl* tc_c_path;

	wxButton* calibrate_button;
	wxButton* pos_reg_button;

	wxButton* ok_button;

	CameraControlFrame(wxWindow* parent);

	void OnCameraControlFrameButtonCalibrate(wxCommandEvent& event);
	void OnCameraControlFrameButtonPositionRegression(wxCommandEvent& event);

	void OnCameraControlFrameButtonOk(wxCommandEvent& event);
	void OnCameraControlFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};
