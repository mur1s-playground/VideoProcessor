#pragma once

#include "ApplicationGraph.h"

void im_show_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class ImShowFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc;

	ImShowFrame(wxWindow* parent);

	void OnImShowFrameButtonOk(wxCommandEvent& event);
	void OnImShowFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};