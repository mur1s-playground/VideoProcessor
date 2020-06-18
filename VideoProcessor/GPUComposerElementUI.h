#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_composer_element_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUComposerElementFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_dx;
	wxTextCtrl* tc_dy;

	wxTextCtrl* tc_scale;

	wxTextCtrl* tc_crop_x1;
	wxTextCtrl* tc_crop_x2;
	wxTextCtrl* tc_crop_y1;
	wxTextCtrl* tc_crop_y2;

	GPUComposerElementFrame(wxWindow* parent);

	void OnGPUComposerElementFrameButtonOk(wxCommandEvent& event);
	void OnGPUComposerElementFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};