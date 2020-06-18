#include "GPUComposerElementUI.h"
#include "GPUComposerElement.h"

#include "MainUI.h"

const string TEXT_VIDEO_SOURCE = "Video Source";

void gpu_composer_element_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_GPU_COMPOSER_ELEMENT;

    agn->name = "GPU Composer Element";

    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_GPU_COMPOSER_ELEMENT, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct gpu_composer_element* gce = (struct gpu_composer_element*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gce->dx));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gce->dy));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gce->crop_x1));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gce->crop_x2));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gce->crop_y1));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gce->crop_y2));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&gce->scale));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gce->width));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gce->height));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&gce->vs_in);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->process = nullptr;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = gpu_composer_element_destroy;

    agn->externalise = gpu_composer_element_externalise;
}


GPUComposerElementFrame::GPUComposerElementFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Shared Memory Buffer")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_dx = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_dx = new wxStaticText(panel, -1, wxT("dx"));
    hbox_dx->Add(st_dx, 0, wxRIGHT, 8);

    tc_dx = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_dx->Add(tc_dx, 1);

    vbox->Add(hbox_dx, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_dy = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_dy = new wxStaticText(panel, -1, wxT("dy"));
    hbox_dy->Add(st_dy, 0, wxRIGHT, 8);

    tc_dy = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_dy->Add(tc_dy, 1);

    vbox->Add(hbox_dy, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_crop_x1 = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_crop_x1 = new wxStaticText(panel, -1, wxT("crop_x1"));
    hbox_crop_x1->Add(st_crop_x1, 0, wxRIGHT, 8);

    tc_crop_x1 = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_crop_x1->Add(tc_crop_x1, 1);

    vbox->Add(hbox_crop_x1, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_crop_x2 = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_crop_x2 = new wxStaticText(panel, -1, wxT("crop_x2"));
    hbox_crop_x2->Add(st_crop_x2, 0, wxRIGHT, 8);

    tc_crop_x2 = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_crop_x2->Add(tc_crop_x2, 1);

    vbox->Add(hbox_crop_x2, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_crop_y1 = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_crop_y1 = new wxStaticText(panel, -1, wxT("crop_y1"));
    hbox_crop_y1->Add(st_crop_y1, 0, wxRIGHT, 8);

    tc_crop_y1 = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_crop_y1->Add(tc_crop_y1, 1);

    vbox->Add(hbox_crop_y1, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_crop_y2 = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_crop_y2 = new wxStaticText(panel, -1, wxT("crop_y2"));
    hbox_crop_y2->Add(st_crop_y2, 0, wxRIGHT, 8);

    tc_crop_y2 = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_crop_y2->Add(tc_crop_y2, 1);

    vbox->Add(hbox_crop_y2, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_scale = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_scale = new wxStaticText(panel, -1, wxT("scale"));
    hbox_scale->Add(st_scale, 0, wxRIGHT, 8);

    tc_scale = new wxTextCtrl(panel, -1, wxT("1.0"));
    hbox_scale->Add(tc_scale, 1);

    vbox->Add(hbox_scale, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &GPUComposerElementFrame::OnGPUComposerElementFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &GPUComposerElementFrame::OnGPUComposerElementFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);

}

void GPUComposerElementFrame::OnGPUComposerElementFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str_dx = tc_dx->GetValue();
    tc_dx->SetValue(wxT("0"));

    wxString str_dy = tc_dy->GetValue();
    tc_dy->SetValue(wxT("0"));

    wxString str_crop_x1 = tc_crop_x1->GetValue();
    tc_crop_x1->SetValue(wxT("0"));
    wxString str_crop_x2 = tc_crop_x2->GetValue();
    tc_crop_x2->SetValue(wxT("0"));
    wxString str_crop_y1 = tc_crop_y1->GetValue();
    tc_crop_y1->SetValue(wxT("0"));
    wxString str_crop_y2 = tc_crop_y2->GetValue();
    tc_crop_y2->SetValue(wxT("0"));

    wxString str_scale = tc_scale->GetValue();
    tc_scale->SetValue(wxT("1.00"));

    struct gpu_composer_element* gce;
    struct application_graph_node* agn;
    if (node_id == -1) {
        gce = new gpu_composer_element();
        gpu_composer_element_init(gce);
    } else {
        agn = ags[node_graph_id]->nodes[node_id];
        gce = (struct gpu_composer_element*)agn->component;
    }
    
    gce->dx = stoi(str_dx.c_str().AsChar());
    gce->dy = stoi(str_dy.c_str().AsChar());

    gce->crop_x1 = stoi(str_crop_x1.c_str().AsChar());
    gce->crop_x2 = stoi(str_crop_x2.c_str().AsChar());
    gce->crop_y1 = stoi(str_crop_y1.c_str().AsChar());
    gce->crop_y2 = stoi(str_crop_y2.c_str().AsChar());

    gce->scale = stof(str_scale.c_str().AsChar());

    if (node_id == -1) {
        agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        gpu_composer_element_ui_graph_init(agn, (application_graph_component)gce, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    }
    myApp->drawPane->Refresh();
}

void GPUComposerElementFrame::OnGPUComposerElementFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_dx->SetValue(wxT("0"));
    tc_dy->SetValue(wxT("0"));
    tc_crop_x1->SetValue(wxT("0"));
    tc_crop_x2->SetValue(wxT("0"));
    tc_crop_y1->SetValue(wxT("0"));
    tc_crop_y2->SetValue(wxT("0"));
    tc_scale->SetValue(wxT("1.0"));
}

void GPUComposerElementFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_composer_element* gc = (struct gpu_composer_element*)agn->component;
        stringstream s_dx;
        s_dx << gc->dx;
        tc_dx->SetValue(wxString(s_dx.str()));

        stringstream s_dy;
        s_dy << gc->dy;
        tc_dy->SetValue(wxString(s_dy.str()));

        stringstream s_crop_x1;
        s_crop_x1 << gc->crop_x1;
        tc_crop_x1->SetValue(wxString(s_crop_x1.str()));

        stringstream s_crop_x2;
        s_crop_x2 << gc->crop_x2;
        tc_crop_x2->SetValue(wxString(s_crop_x2.str()));

        stringstream s_crop_y1;
        s_crop_y1 << gc->crop_y1;
        tc_crop_y1->SetValue(wxString(s_crop_y1.str()));

        stringstream s_crop_y2;
        s_crop_y2 << gc->crop_y2;
        tc_crop_y2->SetValue(wxString(s_crop_y2.str()));

        stringstream s_scale;
        s_scale << gc->scale;
        tc_scale->SetValue(wxString(s_scale.str()));
    }
    wxFrame::Show(true);
}