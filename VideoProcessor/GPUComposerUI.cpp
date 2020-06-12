#include "GPUComposerUI.h"
#include "GPUComposer.h"

#include "MainUI.h"

const string TEXT_GPU_COMPOSER_ELEMENTS = "GPU Composer Elements";
const string TEXT_VIDEO_SOURCE_OUT = "Video Source OUT";

void gpu_composer_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_GPU_COMPOSER;

    agn->name = "GPU Composer";

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct gpu_composer* gc = (struct gpu_composer*)agc;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&gc->name));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_GPU_COMPOSER_ELEMENTS));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_GPU_COMPOSER_ELEMENT, (void*)&gc->gce_in_connector);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE_OUT));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&gc->vs_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));

    agn->process = gpu_composer_loop;
    agn->process_run = false;
    agn->on_input_connect = gpu_composer_on_input_connect;
}

GPUComposerFrame::GPUComposerFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Im Show")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_name = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_name = new wxStaticText(panel, -1, wxT("Name"));
    hbox_name->Add(st_name, 0, wxRIGHT, 8);

    tc = new wxTextCtrl(panel, -1, wxT(""));
    hbox_name->Add(tc, 1);

    vbox->Add(hbox_name, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &GPUComposerFrame::OnGPUComposerFrameButtonOk, this);

    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &GPUComposerFrame::OnGPUComposerFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void GPUComposerFrame::OnGPUComposerFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str = tc->GetValue();
    tc->SetValue(wxT(""));
    struct gpu_composer* gc = new gpu_composer();
    gpu_composer_init(gc, str.c_str().AsChar());
    struct application_graph_node* agn = new application_graph_node();
    gpu_composer_ui_graph_init(agn, (application_graph_component)gc, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
    ags[0]->nodes.push_back(agn);
    myApp->drawPane->Refresh();
}

void GPUComposerFrame::OnGPUComposerFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc->SetValue(wxT(""));
}