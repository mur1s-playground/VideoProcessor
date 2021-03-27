#pragma once

#include "ApplicationGraph.h"

void statistics_3d_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class Statistics3DFrame : public wxFrame {
	int node_graph_id;
	int node_id;
public:
	wxButton* ok_button;

	Statistics3DFrame(wxWindow* parent);

	void OnStatistics3DFrameButtonCalibrate(wxCommandEvent& event);

	void OnStatistics3DFrameButtonOk(wxCommandEvent& event);
	void OnStatistics3DFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};
