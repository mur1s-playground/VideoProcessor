#include "GPUPaletteFilterUI.h"

#include "GPUPaletteFilter.h"

#include "MainUI.h"
#include <fstream>
#include <sstream>

const string TEXT_VIDEO_SOURCE = "Video Source";
const string TEXT_GPU_MEMORY_BUFFER_OUT = "GPU Memory Buffer Out";

void gpu_palette_filter_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_GPU_PALETTE_FILTER;

    agn->name = "GPU Palette Filter";

    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_GPU_PALETTE_FILTER, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct gpu_palette_filter* gpf = (struct gpu_palette_filter*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&gpf->palette_auto_time));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gpf->palette_auto_size));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gpf->palette_auto_bucket_count));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gpf->palette_auto_quantization_size));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_BOOL, (void*)&gpf->device_palette_switch));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gpf->palette_size[0]));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&gpf->palette_size[1]));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    /*
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING_LIST, (void*)&gpf->palette));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    */
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&gpf->vs_in);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_GPU_MEMORY_BUFFER_OUT));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_GPU_MEMORY_BUFFER, (void*)&gpf->gmb_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = gpu_palette_filter_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = gpu_palette_filter_destroy;
    agn->externalise = gpu_palette_filter_externalise;
}

GPUPaletteFilterFrame::GPUPaletteFilterFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("GPU Palette Filter")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_fc = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_fc = new wxStaticText(panel, -1, wxT("Palette auto time"));
    hbox_fc->Add(st_fc, 0, wxRIGHT, 8);

    tc_palette_auto_time = new wxTextCtrl(panel, -1, wxT("10.0"));
    hbox_fc->Add(tc_palette_auto_time, 1);

    vbox->Add(hbox_fc, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_a = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_a = new wxStaticText(panel, -1, wxT("Palette auto size"));
    hbox_a->Add(st_a, 0, wxRIGHT, 8);

    tc_palette_auto_size = new wxTextCtrl(panel, -1, wxT("128"));
    hbox_a->Add(tc_palette_auto_size, 1);

    vbox->Add(hbox_a, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_b = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_b = new wxStaticText(panel, -1, wxT("Palette auto bucket count"));
    hbox_b->Add(st_b, 0, wxRIGHT, 8);

    tc_palette_auto_bucket_count = new wxTextCtrl(panel, -1, wxT("10"));
    hbox_b->Add(tc_palette_auto_bucket_count, 1);

    vbox->Add(hbox_b, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_c = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_c = new wxStaticText(panel, -1, wxT("Palette auto quantization size"));
    hbox_c->Add(st_c, 0, wxRIGHT, 8);

    tc_palette_auto_quantization_size = new wxTextCtrl(panel, -1, wxT("16"));
    hbox_c->Add(tc_palette_auto_quantization_size, 1);

    vbox->Add(hbox_c, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_classes = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_classes = new wxStaticText(panel, wxID_ANY,
        wxT("Palette"));

    hbox_classes->Add(st_classes, 0);
    vbox->Add(hbox_classes, 0, wxLEFT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_classes_t = new wxBoxSizer(wxHORIZONTAL);
    tc_palette = new wxTextCtrl(panel, wxID_ANY, wxString(""),
        wxPoint(-1, -1), wxSize(-1, -1), wxTE_MULTILINE);
    tc_palette->SetMinSize(wxSize(200, 200));

    hbox_classes_t->Add(tc_palette, 1, wxEXPAND);
    vbox->Add(hbox_classes_t, 1, wxLEFT | wxRIGHT | wxEXPAND, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &GPUPaletteFilterFrame::OnGPUPaletteFilterFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &GPUPaletteFilterFrame::OnGPUPaletteFilterFrameButtonClose, this);

    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void GPUPaletteFilterFrame::OnGPUPaletteFilterFrameButtonOk(wxCommandEvent& event) {
    this->Hide();

    struct gpu_palette_filter* gpf;
    if (node_id == -1) {
        gpf = new gpu_palette_filter();
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        gpf = (struct gpu_palette_filter*)agn->component;
    }
    
    float palette_auto_time = stof(string(tc_palette_auto_time->GetValue()));
    int palette_auto_size = stoi(string(tc_palette_auto_size->GetValue()));
    int palette_auto_bucket_count = stoi(string(tc_palette_auto_bucket_count->GetValue()));
    int palette_auto_quantization_size = stoi(string(tc_palette_auto_quantization_size->GetValue()));

    if (palette_auto_time == 0.0f) {
        wxString classes_l = tc_palette->GetValue();
        int start = 0;
        int end = classes_l.find_first_of(",", start);
        while (end != std::string::npos) {
            gpf->palette.push_back(stof(string(classes_l.substr(start, end - start).c_str())));
            start = end + 1;
            end = classes_l.find_first_of(",", start);
        }
        gpf->palette.push_back(stof(string(classes_l.substr(start, end - start).c_str())));
    }
    
    if (node_id == -1) {
        gpu_palette_filter_init(gpf, palette_auto_time, palette_auto_size, palette_auto_bucket_count, palette_auto_quantization_size);
        application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        gpu_palette_filter_ui_graph_init(agn, (application_graph_component)gpf, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        gpf->palette_auto_time = palette_auto_time;
        gpf->palette_auto_size = palette_auto_size;
        gpf->palette_auto_bucket_count = palette_auto_bucket_count;
        gpf->palette_auto_quantization_size = palette_auto_quantization_size;
        gpf->palette_auto_timer = 0.0f;
    }
    myApp->drawPane->Refresh();
}

void GPUPaletteFilterFrame::OnGPUPaletteFilterFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_palette->SetValue("");
}

void GPUPaletteFilterFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_palette_filter* gpf = (struct gpu_palette_filter*)agn->component;

        wxString auto_time;
        auto_time << gpf->palette_auto_time;
        tc_palette_auto_time->SetValue(auto_time);

        wxString auto_size;
        auto_size << gpf->palette_auto_size;
        tc_palette_auto_size->SetValue(auto_size);

        wxString auto_bucket_count;
        auto_bucket_count << gpf->palette_auto_bucket_count;
        tc_palette_auto_bucket_count->SetValue(auto_bucket_count);

        wxString auto_quantization_size;
        auto_quantization_size << gpf->palette_auto_quantization_size;
        tc_palette_auto_quantization_size->SetValue(auto_quantization_size);

        stringstream s_classes;
        for (int i = 0; i < gpf->palette.size(); i++) {
            if (i > 0) {
                s_classes << ",";
            }
            s_classes << gpf->palette[i];
        }

        tc_palette->SetValue(wxString(s_classes.str().c_str()));
    }
    wxFrame::Show(true);
}