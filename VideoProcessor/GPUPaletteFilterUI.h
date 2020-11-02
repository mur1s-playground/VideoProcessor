#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_palette_filter_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUPaletteFilterFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_palette;

	GPUPaletteFilterFrame(wxWindow* parent);

	void OnGPUPaletteFilterFrameButtonOk(wxCommandEvent& event);
	void OnGPUPaletteFilterFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};