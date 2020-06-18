#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_gaussian_blur_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUGaussianBlurFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_ks;
	wxTextCtrl* tc_a;
	wxTextCtrl* tc_b;
	wxTextCtrl* tc_c;

	GPUGaussianBlurFrame(wxWindow* parent);

	void OnGPUGaussianBlurFrameButtonOk(wxCommandEvent& event);
	void OnGPUGaussianBlurFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};