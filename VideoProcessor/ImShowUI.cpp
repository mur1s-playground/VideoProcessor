#include "ImShowUI.h"

#include "ImShow.h"
#include "VideoSource.h"

#include "MainUI.h"

const string TEXT_VIDEO_SOURCE = "Video Source";

void im_show_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_IM_SHOW;

    agn->name = "Im Show";
    
    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct im_show* is = (struct im_show*)agc;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&is->name));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&is->vs);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->process = im_show_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = im_show_destroy;
    agn->externalise = im_show_externalise;
}

ImShowFrame::ImShowFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Im Show")) {

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
    ok_button->Bind(wxEVT_BUTTON, &ImShowFrame::OnImShowFrameButtonOk, this);

    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &ImShowFrame::OnImShowFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void ImShowFrame::OnImShowFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str = tc->GetValue();
    tc->SetValue(wxT(""));
    if (node_id == -1) {
        struct im_show* is = new im_show();
        im_show_init(is, str.c_str().AsChar());
        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        im_show_ui_graph_init(agn, (application_graph_component)is, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct im_show* is = (struct im_show*)agn->component;
        is->name = str.c_str().AsChar();
    }
    myApp->drawPane->Refresh();
}

void ImShowFrame::OnImShowFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc->SetValue(wxT(""));
}

void ImShowFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct im_show* is = (struct im_show*)agn->component;
        tc->SetValue(wxString(is->name));
    }
    wxFrame::Show(true);
}