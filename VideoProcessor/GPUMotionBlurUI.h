#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_motion_blur_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUMotionBlurFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_fc;

	GPUMotionBlurFrame(wxWindow* parent);

	void OnGPUMotionBlurFrameButtonOk(wxCommandEvent& event);
	void OnGPUMotionBlurFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};