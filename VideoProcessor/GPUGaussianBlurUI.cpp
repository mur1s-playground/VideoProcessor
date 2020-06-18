#include "GPUGaussianBlurUI.h"

#include "GPUGaussianBlur.h"

#include "MainUI.h"

const string TEXT_VIDEO_SOURCE = "Video Source";
const string TEXT_GPU_MEMORY_BUFFER_OUT = "GPU Memory Buffer Out";

void gpu_gaussian_blur_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_GPU_GAUSSIAN_BLUR;

    agn->name = "GPU Gaussian Blur";

    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_GPU_GAUSSIAN_BLUR, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct gpu_gaussian_blur* mb = (struct gpu_gaussian_blur*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&mb->kernel_size));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&mb->a));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&mb->b));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&mb->c));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&mb->vs_in);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_GPU_MEMORY_BUFFER_OUT));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_GPU_MEMORY_BUFFER, (void*)&mb->gmb_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));

    agn->process = gpu_gaussian_blur_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = gpu_gaussian_blur_destroy;
    agn->externalise = gpu_gaussian_blur_externalise;
}

GPUGaussianBlurFrame::GPUGaussianBlurFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("GPU Gaussian Blur")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_fc = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_fc = new wxStaticText(panel, -1, wxT("Kernel Size"));
    hbox_fc->Add(st_fc, 0, wxRIGHT, 8);

    tc_ks = new wxTextCtrl(panel, -1, wxT("25"));
    hbox_fc->Add(tc_ks, 1);

    vbox->Add(hbox_fc, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_a = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_a = new wxStaticText(panel, -1, wxT("a"));
    hbox_a->Add(st_a, 0, wxRIGHT, 8);

    tc_a = new wxTextCtrl(panel, -1, wxT("1.0"));
    hbox_a->Add(tc_a, 1);

    vbox->Add(hbox_a, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_b = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_b = new wxStaticText(panel, -1, wxT("b"));
    hbox_b->Add(st_b, 0, wxRIGHT, 8);

    tc_b = new wxTextCtrl(panel, -1, wxT("0.0"));
    hbox_b->Add(tc_b, 1);

    vbox->Add(hbox_b, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_c = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_c = new wxStaticText(panel, -1, wxT("c"));
    hbox_c->Add(st_c, 0, wxRIGHT, 8);

    tc_c = new wxTextCtrl(panel, -1, wxT("25.0"));
    hbox_c->Add(tc_c, 1);

    vbox->Add(hbox_c, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &GPUGaussianBlurFrame::OnGPUGaussianBlurFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &GPUGaussianBlurFrame::OnGPUGaussianBlurFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void GPUGaussianBlurFrame::OnGPUGaussianBlurFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str = tc_ks->GetValue();
    tc_ks->SetValue(wxT("25"));

    wxString str_a = tc_a->GetValue();
    tc_a->SetValue(wxT("1.0"));

    wxString str_b = tc_b->GetValue();
    tc_b->SetValue(wxT("0.0"));

    wxString str_c = tc_c->GetValue();
    tc_c->SetValue(wxT("25.0"));

    if (node_id == -1) {
        struct gpu_gaussian_blur* mb = new gpu_gaussian_blur();
        gpu_gaussian_blur_init(mb, stoi(str.c_str().AsChar()), stof(str_a.c_str().AsChar()), stof(str_b.c_str().AsChar()), stof(str_c.c_str().AsChar()));

        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        gpu_gaussian_blur_ui_graph_init(agn, (application_graph_component)mb, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_gaussian_blur* mb = (struct gpu_gaussian_blur*)agn->component;
        gpu_gaussian_blur_edit(mb, stoi(str.c_str().AsChar()), stof(str_a.c_str().AsChar()), stof(str_b.c_str().AsChar()), stof(str_c.c_str().AsChar()));
    }
    myApp->drawPane->Refresh();
}

void GPUGaussianBlurFrame::OnGPUGaussianBlurFrameButtonClose(wxCommandEvent& event) {
    this->Hide();   
    tc_ks->SetValue(wxT("25"));  
    tc_a->SetValue(wxT("1.0"));  
    tc_b->SetValue(wxT("0.0")); 
    tc_c->SetValue(wxT("25.0"));
}

void GPUGaussianBlurFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_gaussian_blur* mb = (struct gpu_gaussian_blur*)agn->component;
        stringstream s_sw;
        s_sw << mb->kernel_size;
        tc_ks->SetValue(wxString(s_sw.str()));

        stringstream s_a;
        s_a << mb->a;
        tc_a->SetValue(wxString(s_a.str()));

        stringstream s_b;
        s_b << mb->b;
        tc_b->SetValue(wxString(s_b.str()));

        stringstream s_c;
        s_c << mb->c;
        tc_c->SetValue(wxString(s_c.str()));
    }
    wxFrame::Show(true);
}