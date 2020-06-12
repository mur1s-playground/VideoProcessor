#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void shared_memory_buffer_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class SharedMemoryBufferFrame : public wxFrame {

public:
	wxTextCtrl* tc_name;

	wxTextCtrl* tc_size;

	wxTextCtrl* tc_slots;

	SharedMemoryBufferFrame(wxWindow* parent);

	void OnSharedMemoryBufferFrameButtonOk(wxCommandEvent& event);
	void OnSharedMemoryBufferFrameButtonClose(wxCommandEvent& event);
};