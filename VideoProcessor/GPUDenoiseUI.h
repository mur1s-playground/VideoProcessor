#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_denoise_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUDenoiseFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_searchwindow;

	wxTextCtrl* tc_region;

	wxTextCtrl* tc_filtering;

	GPUDenoiseFrame(wxWindow* parent);

	void OnGPUDenoiseButtonOk(wxCommandEvent& event);
	void OnGPUDenoiseButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};
