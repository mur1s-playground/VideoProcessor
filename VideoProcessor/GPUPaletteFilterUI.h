#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_palette_filter_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUPaletteFilterFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_palette_auto_time;
	wxTextCtrl* tc_palette_auto_size;
	wxTextCtrl* tc_palette_auto_bucket_count;
	wxTextCtrl* tc_palette_auto_quantization_size;

	wxTextCtrl* tc_palette;

	GPUPaletteFilterFrame(wxWindow* parent);

	void OnGPUPaletteFilterFrameButtonOk(wxCommandEvent& event);
	void OnGPUPaletteFilterFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};