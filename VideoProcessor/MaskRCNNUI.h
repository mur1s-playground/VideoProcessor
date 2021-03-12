#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void mask_rcnn_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class MaskRCNNFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	string classes;

	wxTextCtrl* tc_confthres;

	wxTextCtrl* tc_maskthres;

	wxTextCtrl* tc_classes;

	wxTextCtrl* tc_scale;

	wxChoice* ch_draw_box;
	wxChoice* ch_draw_mask;

	MaskRCNNFrame(wxWindow* parent);

	void OnMaskRCNNFrameButtonOk(wxCommandEvent& event);
	void OnMaskRCNNFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};