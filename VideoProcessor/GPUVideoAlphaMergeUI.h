#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_video_alpha_merge_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUVideoAlphaMergeFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc;
	wxTextCtrl* tc_d;

	GPUVideoAlphaMergeFrame(wxWindow* parent);

	void OnGPUVideoAlphaMergeFrameButtonOk(wxCommandEvent& event);
	void OnGPUVideoAlphaMergeFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};

