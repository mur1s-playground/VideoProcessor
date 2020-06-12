#include "GPUDenoiseUI.h"

#include "GPUDenoise.h"

#include "MainUI.h"

const string TEXT_VIDEO_SOURCE = "Video Source";
const string TEXT_GPU_MEMORY_BUFFER_OUT = "GPU Memory Buffer Out";

void gpu_denoise_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_GPU_DENOISE;

    agn->name = "GPU Denoise";

    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_GPU_DENOISE, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct gpu_denoise* gd = (struct gpu_denoise*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gd->search_window_size));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gd->region_size));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&gd->filtering_param));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&gd->vs_in);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_GPU_MEMORY_BUFFER_OUT));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_GPU_MEMORY_BUFFER, (void*)&gd->gmb_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));

    agn->process = gpu_denoise_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
}

GPUDenoiseFrame::GPUDenoiseFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("GPU Denoise")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_search = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_search = new wxStaticText(panel, -1, wxT("Search Window Size"));
    hbox_search->Add(st_search, 0, wxRIGHT, 8);

    tc_searchwindow = new wxTextCtrl(panel, -1, wxT("21"));
    hbox_search->Add(tc_searchwindow, 1);

    vbox->Add(hbox_search, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_region = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_region = new wxStaticText(panel, -1, wxT("Region Size"));
    hbox_region->Add(st_region, 0, wxRIGHT, 8);

    tc_region = new wxTextCtrl(panel, -1, wxT("7"));
    hbox_region->Add(tc_region, 1);

    vbox->Add(hbox_region, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_filtering = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_filtering = new wxStaticText(panel, -1, wxT("Filtering Parameter"));
    hbox_filtering->Add(st_filtering, 0, wxRIGHT, 8);

    tc_filtering = new wxTextCtrl(panel, -1, wxT("3.0"));
    hbox_filtering->Add(tc_filtering, 1);

    vbox->Add(hbox_filtering, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &GPUDenoiseFrame::OnGPUDenoiseButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &GPUDenoiseFrame::OnGPUDenoiseButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void GPUDenoiseFrame::OnGPUDenoiseButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str = tc_searchwindow->GetValue();
    tc_searchwindow->SetValue(wxT("21"));

    wxString str_region = tc_region->GetValue();
    tc_region->SetValue(wxT("7"));

    wxString str_filtering = tc_filtering->GetValue();
    tc_filtering->SetValue(wxT("3.0"));

    struct gpu_denoise* gd = new gpu_denoise();
    gpu_denoise_init(gd, stoi(str.c_str().AsChar()), stoi(str_region.c_str().AsChar()), stof(str_filtering.c_str().AsChar()));

    struct application_graph_node* agn = new application_graph_node();
    gpu_denoise_ui_graph_init(agn, (application_graph_component)gd, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
    ags[0]->nodes.push_back(agn);
    myApp->drawPane->Refresh();
}

void GPUDenoiseFrame::OnGPUDenoiseButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_searchwindow->SetValue(wxT("21"));
    tc_region->SetValue(wxT("7"));
    tc_filtering->SetValue(wxT("3.0"));
}