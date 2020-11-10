#include "AudioSourceUI.h"

#include <sstream>
#include <vector>

#include "AudioSource.h"

#include "Logger.h"

#include "MainUI.h"
#include "wx/wx.h"

using namespace std;

const string TEXT_SHARED_MEMORY_BUFFER = "Memory Buffer";
const string TEXT_GPU_MEMORY_BUFFER = "GPU Memory Buffer";

void audio_source_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_AUDIO_SOURCE;

    agn->name = "Audio Source";
    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_AUDIO_SOURCE, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct audio_source* as = (struct audio_source*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&as->device_id));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_SHARED_MEMORY_BUFFER));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_SHARED_MEMORY_BUFFER, (void*)&as->smb);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&as->smb_size_req));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_GPU_MEMORY_BUFFER));

    pair<enum application_graph_component_type, void*> gpu_in = pair<enum application_graph_component_type, void*>(AGCT_GPU_MEMORY_BUFFER, (void*)&as->gmb);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, gpu_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = audio_source_loop;
    agn->process_run = false;

    agn->on_input_connect = audio_source_on_input_connect;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = audio_source_destory;
    agn->externalise = audio_source_externalise;
}

AudioSourceFrame::AudioSourceFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Audio Source")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_path = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_path = new wxStaticText(panel, -1, wxT("DeviceId"));
    hbox_path->Add(st_path, 0, wxRIGHT, 8);

    tc_device_id = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_path->Add(tc_device_id, 1);

    vbox->Add(hbox_path, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_channels = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_channels = new wxStaticText(panel, -1, wxT("Channels"));
    hbox_channels->Add(st_channels, 0, wxRIGHT, 8);

    tc_channels = new wxTextCtrl(panel, -1, wxT("2"));
    hbox_channels->Add(tc_channels, 1);

    vbox->Add(hbox_channels, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_sps = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_sps = new wxStaticText(panel, -1, wxT("Samples per Sec"));
    hbox_sps->Add(st_sps, 0, wxRIGHT, 8);

    tc_samples_per_sec = new wxTextCtrl(panel, -1, wxT("44100"));
    hbox_sps->Add(tc_samples_per_sec, 1);

    vbox->Add(hbox_sps, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    
    wxBoxSizer* hbox_bps = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_bps = new wxStaticText(panel, -1, wxT("Bits per Sample"));
    hbox_bps->Add(st_bps, 0, wxRIGHT, 8);

    tc_bits_per_sample = new wxTextCtrl(panel, -1, wxT("8"));
    hbox_bps->Add(tc_bits_per_sample, 1);

    vbox->Add(hbox_bps, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_copy = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_copy = new wxStaticText(panel, -1, wxT("Copy to GPU"));
    hbox_copy->Add(st_copy, 0, wxRIGHT, 8);

    tc_copy_to_gmb = new wxTextCtrl(panel, -1, wxT("1"));
    hbox_copy->Add(tc_copy_to_gmb, 1);

    vbox->Add(hbox_copy, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &AudioSourceFrame::OnAudioSourceFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &AudioSourceFrame::OnAudioSourceFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void AudioSourceFrame::OnAudioSourceFrameButtonOk(wxCommandEvent& event) {
    this->Hide();

    int device_id = stoi(tc_device_id->GetValue().c_str().AsChar());
    int channels = stoi(tc_channels->GetValue().c_str().AsChar());
    int samples_per_sec = stoi(tc_samples_per_sec->GetValue().c_str().AsChar());
    int bits_per_sample = stoi(tc_bits_per_sample->GetValue().c_str().AsChar());
    bool copy_to_gmb = stoi(tc_copy_to_gmb->GetValue().c_str().AsChar()) == 1;
   
    struct audio_source* as;
    if (node_id == -1) {
        as = new audio_source();
        audio_source_init(as, device_id, channels, samples_per_sec, bits_per_sample, copy_to_gmb);

        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        audio_source_ui_graph_init(agn, (application_graph_component)as, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        as = (struct audio_source*)agn->component;
        //audio_source_edit();
    }
    myApp->drawPane->Refresh();
}

void AudioSourceFrame::OnAudioSourceFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_device_id->SetValue(wxT("0"));
    tc_channels->SetValue(wxT("2"));
    tc_samples_per_sec->SetValue(wxT("44100"));
    tc_bits_per_sample->SetValue(wxT("8"));
    tc_copy_to_gmb->SetValue(wxT("1"));
}

void AudioSourceFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct audio_source* as = (struct audio_source*)agn->component;

        stringstream s_device_id;
        s_device_id << as->device_id;
        tc_device_id->SetValue(wxString(s_device_id.str()));

        stringstream s_channels;
        s_channels << as->wave_format.nChannels;
        tc_channels->SetValue(wxString(s_channels.str()));

        stringstream s_samples_per_sec;
        s_samples_per_sec << as->wave_format.nSamplesPerSec;
        tc_samples_per_sec->SetValue(wxString(s_samples_per_sec.str()));

        stringstream s_bits_per_sample;
        s_bits_per_sample << as->wave_format.wBitsPerSample;
        tc_bits_per_sample->SetValue(wxString(s_bits_per_sample.str()));

        if (as->copy_to_gmb) {
            tc_copy_to_gmb->SetValue(wxString("1"));
        } else {
            tc_copy_to_gmb->SetValue(wxString("0"));
        }
    }
    wxFrame::Show(true);
}