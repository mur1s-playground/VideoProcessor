#include "ApplicationGraphNodeSettingsUI.h"

#include "MainUI.h"

#include <sstream>

ApplicationGraphNodeSettingsFrame::ApplicationGraphNodeSettingsFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Node Settings"), wxPoint(50, 50), wxSize(260, 200)) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_tps_target = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_tps_target = new wxStaticText(panel, -1, wxT("TPS Target"));
    hbox_tps_target->Add(st_tps_target, 0, wxRIGHT, 8);

    tc_tps_target = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_tps_target->Add(tc_tps_target, 1);

    vbox->Add(hbox_tps_target, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &ApplicationGraphNodeSettingsFrame::OnApplicationGraphNodeSettingsFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &ApplicationGraphNodeSettingsFrame::OnApplicationGraphNodeSettingsFrameButtonClose, this);

    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void ApplicationGraphNodeSettingsFrame::OnApplicationGraphNodeSettingsFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str_d = tc_tps_target->GetValue();
    tc_tps_target->SetValue(wxT("30"));

    struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
    agn->process_tps_balancer.tps_target = stoi(str_d.c_str().AsChar());
    
    myApp->drawPane->Refresh();
}

void ApplicationGraphNodeSettingsFrame::OnApplicationGraphNodeSettingsFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_tps_target->SetValue(wxT("30"));
}

void ApplicationGraphNodeSettingsFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    
    struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];

    stringstream s_d;
    s_d << agn->process_tps_balancer.tps_target;
    tc_tps_target->SetValue(wxString(s_d.str()));

    wxFrame::Show(true);
}