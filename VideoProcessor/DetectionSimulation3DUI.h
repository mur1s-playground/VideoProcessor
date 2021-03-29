#pragma once

#include "ApplicationGraph.h"

void detection_simulation_3d_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class DetectionSimulation3DFrame : public wxFrame {
	int node_graph_id;
	int node_id;
public:
	wxButton* ok_button;

	DetectionSimulation3DFrame(wxWindow* parent);

	void OnDetectionSimulation3DFrameButtonOk(wxCommandEvent& event);
	void OnDetectionSimulation3DFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};
