#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_motion_blur_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUMotionBlurFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxBoxSizer* vbox;

	wxTextCtrl* tc_fc;

	wxChoice* ch_dt;

	wxBoxSizer* hbox_wc;
	wxTextCtrl* tc_wc;

	wxBoxSizer* hbox_c;
	wxTextCtrl* tc_c;

	wxBoxSizer* hbox_p;
	wxTextCtrl* tc_p;

	GPUMotionBlurFrame(wxWindow* parent);

	void OnWeightDistChange(wxCommandEvent& event);

	void OnGPUMotionBlurFrameButtonOk(wxCommandEvent& event);
	void OnGPUMotionBlurFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};