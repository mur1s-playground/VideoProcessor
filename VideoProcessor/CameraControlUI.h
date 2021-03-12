#pragma once

#pragma once

#include "ApplicationGraph.h"

void camera_control_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class CameraControlFrame : public wxFrame {
	int node_graph_id;
	int node_id;
public:
	wxButton* ok_button;

	CameraControlFrame(wxWindow* parent);

	void OnCameraControlFrameButtonOk(wxCommandEvent& event);
	void OnCameraControlFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};
