#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_green_screen_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUGreenScreenFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_r;
	wxTextCtrl* tc_g;
	wxTextCtrl* tc_b;
	wxTextCtrl* tc_threshold;

	GPUGreenScreenFrame(wxWindow* parent);

	void OnGPUGreenScreenFrameButtonOk(wxCommandEvent& event);
	void OnGPUGreenScreenFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};