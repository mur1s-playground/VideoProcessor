#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_edge_filter_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUEdgeFilterFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_amplify;

	GPUEdgeFilterFrame(wxWindow* parent);

	void OnGPUEdgeFilterButtonOk(wxCommandEvent& event);
	void OnGPUEdgeFilterButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};
