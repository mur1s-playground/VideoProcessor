#include "SharedMemoryBufferUI.h"

#include "SharedMemoryBuffer.h"
#include <sstream>

#include "MainUI.h"

void shared_memory_buffer_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_SHARED_MEMORY_BUFFER;

    agn->name = "Shared Memory Buffer";

    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_SHARED_MEMORY_BUFFER, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;
    
    struct shared_memory_buffer* smb = (struct shared_memory_buffer*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&smb->name));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&smb->size));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&smb->slots));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_BOOL, (void*)&smb->error));

    agn->process = nullptr;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = shared_memory_buffer_destroy;
    agn->externalise = shared_memory_buffer_externalise;
}

SharedMemoryBufferFrame::SharedMemoryBufferFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Shared Memory Buffer")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_name = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_name = new wxStaticText(panel, -1, wxT("Name"));
    hbox_name->Add(st_name, 0, wxRIGHT, 8);

    tc_name = new wxTextCtrl(panel, -1, wxT(""));
    hbox_name->Add(tc_name, 1);

    vbox->Add(hbox_name, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_size = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_size = new wxStaticText(panel, -1, wxT("Size"));
    hbox_size->Add(st_size, 0, wxRIGHT, 8);

    tc_size = new wxTextCtrl(panel, -1, wxT(""));
    hbox_size->Add(tc_size, 1);

    vbox->Add(hbox_size, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_slots = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_slots = new wxStaticText(panel, -1, wxT("Slots"));
    hbox_slots->Add(st_slots, 0, wxRIGHT, 8);

    tc_slots = new wxTextCtrl(panel, -1, wxT(""));
    hbox_slots->Add(tc_slots, 1);

    vbox->Add(hbox_slots, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
    
    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &SharedMemoryBufferFrame::OnSharedMemoryBufferFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &SharedMemoryBufferFrame::OnSharedMemoryBufferFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);
    
    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);

}

void SharedMemoryBufferFrame::OnSharedMemoryBufferFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str = tc_name->GetValue();
    tc_name->SetValue(wxT(""));

    wxString str_size = tc_size->GetValue();
    tc_size->SetValue(wxT(""));

    wxString str_slots = tc_slots->GetValue();
    tc_slots->SetValue(wxT(""));

    if (node_id == -1) {
        struct shared_memory_buffer* smb = new shared_memory_buffer();
        shared_memory_buffer_init(smb, str.c_str().AsChar(), stoi(str_size.c_str().AsChar()), stoi(str_slots.c_str().AsChar()), sizeof(int));

        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        shared_memory_buffer_ui_graph_init(agn, (application_graph_component)smb, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct shared_memory_buffer* smb = (struct shared_memory_buffer*)agn->component;

        shared_memory_buffer_edit(smb, str.c_str().AsChar(), stoi(str_size.c_str().AsChar()), stoi(str_slots.c_str().AsChar()), sizeof(int));
    }
    myApp->drawPane->Refresh();
}

void SharedMemoryBufferFrame::OnSharedMemoryBufferFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_name->SetValue(wxT(""));
    tc_size->SetValue(wxT(""));
    tc_slots->SetValue(wxT(""));
}

void SharedMemoryBufferFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct shared_memory_buffer* smb = (struct shared_memory_buffer*)agn->component;

        tc_name->SetValue(wxString(smb->name));

        stringstream s_size;
        s_size << smb->size;
        tc_size->SetValue(wxString(s_size.str()));

        stringstream s_slots;
        s_slots << smb->slots;
        tc_slots->SetValue(wxString(s_slots.str()));
    }
    wxFrame::Show(true);
}