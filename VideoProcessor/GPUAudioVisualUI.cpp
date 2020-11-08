#include "GPUAudioVisualUI.h"

#include "GPUAudioVisual.h"

#include "MainUI.h"
#include <fstream>
#include <sstream>

const string TEXT_VIDEO_SOURCE_OUT = "Video Source Out";
const string TEXT_GPU_MEMORY_BUFFER_IN = "GPU Memory Buffer In";

void gpu_audiovisual_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_GPU_AUDIOVISUAL;

    agn->name = "GPU AudioVisual";

    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_GPU_AUDIOVISUAL, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct gpu_audiovisual* gav = (struct gpu_audiovisual*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    /*
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&gpf->palette_auto_time));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gpf->palette_auto_size));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gpf->palette_auto_bucket_count));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gpf->palette_auto_quantization_size));
    */
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    /*
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_BOOL, (void*)&gpf->device_palette_switch));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gpf->palette_size[0]));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gpf->palette_size[1]));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&gpf->palette_auto_timer));
    */
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    /*
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING_LIST, (void*)&gpf->palette));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    */
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_GPU_MEMORY_BUFFER_IN));
    pair<enum application_graph_component_type, void*> gmb_in = pair<enum application_graph_component_type, void*>(AGCT_GPU_MEMORY_BUFFER, (void*)&gav->gmb_in);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, gmb_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE_OUT));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&gav->vs_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = gpu_audiovisual_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = gpu_audiovisual_destroy;
    agn->externalise = gpu_audiovisual_externalise;
}

GPUAudioVisualFrame::GPUAudioVisualFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("GPU AudioVisual")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_fc = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_fc = new wxStaticText(panel, -1, wxT("Name"));
    hbox_fc->Add(st_fc, 0, wxRIGHT, 8);

    tc_name = new wxTextCtrl(panel, -1, wxT("audiovis"));
    hbox_fc->Add(tc_name, 1);

    vbox->Add(hbox_fc, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &GPUAudioVisualFrame::OnGPUAudioVisualFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &GPUAudioVisualFrame::OnGPUAudioVisualFrameButtonClose, this);

    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void GPUAudioVisualFrame::OnGPUAudioVisualFrameButtonOk(wxCommandEvent& event) {
    this->Hide();

    struct gpu_audiovisual* gav;
    if (node_id == -1) {
        gav = new gpu_audiovisual();
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        gav = (struct gpu_audiovisual*)agn->component;
    }

    wxString str = tc_name->GetValue();
    
    if (node_id == -1) {
        vector<string> av_tmp;
        gpu_audiovisual_init(gav, tc_name->GetValue().c_str().AsChar(), av_tmp);

        application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        gpu_audiovisual_ui_graph_init(agn, (application_graph_component)gav, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        /*
        gpf->palette_auto_time = palette_auto_time;
        gpf->palette_auto_size = palette_auto_size;
        gpf->palette_auto_bucket_count = palette_auto_bucket_count;
        gpf->palette_auto_quantization_size = palette_auto_quantization_size;
        gpf->palette_auto_timer = 0.0f;
        gpu_palette_filter_edit(gpf, palette_auto_time, palette_auto_size, palette_auto_bucket_count, palette_auto_quantization_size);
        */
    }
    myApp->drawPane->Refresh();
}

void GPUAudioVisualFrame::OnGPUAudioVisualFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_name->SetValue("audiovis");
}

void GPUAudioVisualFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_audiovisual* gav = (struct gpu_audiovisual*)agn->component;

        wxString name;
        name << gav->name;
        tc_name->SetValue(name);
    }
    wxFrame::Show(true);
}