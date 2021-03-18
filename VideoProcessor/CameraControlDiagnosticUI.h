#pragma once

#include "ApplicationGraph.h"

void camera_control_diagnostic_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class CameraControlDiagnosticFrame : public wxFrame {
	int node_graph_id;
	int node_id;
public:
	
	wxButton* ok_button;

	CameraControlDiagnosticFrame(wxWindow* parent);

	void OnCameraControlDiagnosticFrameButtonOk(wxCommandEvent& event);
	void OnCameraControlDiagnosticFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};
