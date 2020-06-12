#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void gpu_memory_buffer_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUMemoryBufferFrame : public wxFrame {

public:
	wxTextCtrl* tc_name;

	wxTextCtrl* tc_size;

	wxTextCtrl* tc_slots;

	GPUMemoryBufferFrame(wxWindow* parent);

	void OnGPUMemoryBufferFrameButtonOk(wxCommandEvent& event);
	void OnGPUMemoryBufferFrameButtonClose(wxCommandEvent& event);
};