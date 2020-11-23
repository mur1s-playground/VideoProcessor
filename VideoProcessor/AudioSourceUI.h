#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void audio_source_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class AudioSourceFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxChoice* ch_device;
	wxArrayString device_choices;

	wxChoice* ch_channels;
	wxArrayString channels_choices;

	wxChoice* ch_samples_per_sec;
	wxArrayString sps_choices;

	wxTextCtrl* tc_bits_per_sample;

	wxChoice* ch_copy_to_gmb;

	void InitAudioDevices();
	void UpdateAvailableAudioChannels(int device_id);

	AudioSourceFrame(wxWindow* parent);

	void OnAudioSourceDeviceChange(wxCommandEvent& event);

	void OnAudioSourceFrameButtonOk(wxCommandEvent& event);
	void OnAudioSourceFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};