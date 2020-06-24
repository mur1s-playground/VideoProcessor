#include "MainUI.h"

#include "VideoSource.h"
#include "VideoSourceUI.h"

#include "SharedMemoryBuffer.h"
#include "SharedMemoryBufferUI.h"

#include "MaskRCNN.h"
#include "MaskRCNNUI.h"

#include "ImShow.h"
#include "ImShowUI.h"

#include "GPUMemoryBuffer.h"
#include "GPUMemoryBufferUI.h"

#include "CUDAStreamHandler.h"

#include "UIManager.h"

#include "Logger.h"

MyApp *myApp;

IMPLEMENT_APP(MyApp)

MainFrame::MainFrame(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style, const wxString& name) : wxFrame(parent, id, title, pos, size, style, name) {
    m_pMenuBar = new wxMenuBar();
    m_pFileMenu = new wxMenu();
    m_pFileMenu->Append(MENUBAR_ID_OPEN, _T("Open"));
    m_pFileMenu->Append(MENUBAR_ID_SAVE, _T("Save"));
    m_pFileMenu->AppendSeparator();
    m_pFileMenu->Append(MENUBAR_ID_QUIT, _T("Quit"));
    m_pMenuBar->Append(m_pFileMenu, _T("File"));

    SetMenuBar(m_pMenuBar);

    Bind(wxEVT_COMMAND_MENU_SELECTED, &MainFrame::OnMenuBarSelected, this, MENUBAR_ID_OPEN, MENUBAR_ID_QUIT);
}

void MainFrame::OnMenuBarSelected(wxCommandEvent& event) {
    switch (event.GetId()) {
        case MENUBAR_ID_OPEN:
            ags.clear();
            application_graph_load("./save", "unnamedgraph");
            break;
        case MENUBAR_ID_SAVE:
            application_graph_save("./save", "unnamedgraph");
            break;
        case MENUBAR_ID_QUIT:
            break;
        default:
            break;
    }
}

bool MyApp::OnInit()
{
    cuda_stream_handler_init();

    myApp = this;

    struct application_graph* ag = new application_graph();
    ag->name = "unnamed graph";
    ags.push_back(ag);

    wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);

    frame = new MainFrame((wxFrame*)NULL, -1, wxT("VideoProcessor"), wxPoint(50, 50), wxSize(800, 600));

    drawPane = new BasicDrawPane((wxFrame*)frame);
    sizer->Add(drawPane, 1, wxEXPAND);

    frame->SetSizer(sizer);
    frame->SetAutoLayout(true);

    frame->Show();

    return true;
}

BEGIN_EVENT_TABLE(BasicDrawPane, wxPanel)
// some useful events
 EVT_MOTION(BasicDrawPane::mouseMoved)
 EVT_LEFT_DOWN(BasicDrawPane::mouseDown)
 EVT_LEFT_UP(BasicDrawPane::mouseReleased)
 EVT_RIGHT_DOWN(BasicDrawPane::rightClick)
    
 //EVT_LEAVE_WINDOW(BasicDrawPane::mouseLeftWindow)
 EVT_KEY_DOWN(BasicDrawPane::keyPressed)
 EVT_KEY_UP(BasicDrawPane::keyReleased)
 /*
 EVT_MOUSEWHEEL(BasicDrawPane::mouseWheelMoved)
 */

 // catch paint events
    EVT_PAINT(BasicDrawPane::paintEvent)

    END_EVENT_TABLE()


    // some useful events
    /*
     void BasicDrawPane::mouseMoved(wxMouseEvent& event) {}
     void BasicDrawPane::mouseDown(wxMouseEvent& event) {}
     void BasicDrawPane::mouseWheelMoved(wxMouseEvent& event) {}
     void BasicDrawPane::mouseReleased(wxMouseEvent& event) {}
     void BasicDrawPane::rightClick(wxMouseEvent& event) {}
     void BasicDrawPane::mouseLeftWindow(wxMouseEvent& event) {}
     void BasicDrawPane::keyPressed(wxKeyEvent& event) {}
     void BasicDrawPane::keyReleased(wxKeyEvent& event) {}
     */

    BasicDrawPane::BasicDrawPane(wxFrame* parent) :
    wxPanel(parent)
{
    Bind(wxEVT_COMMAND_MENU_SELECTED, &BasicDrawPane::OnContextMenuSelected, this, MENU_ID_SHARED_MEMORY_BUFFER, MENU_ID_START_NODE);
}

bool move_node_started = false;
void BasicDrawPane::mouseMoved(wxMouseEvent& event) {
    const wxPoint pt = wxGetMousePosition();
    mouse_position_x = pt.x - this->GetScreenPosition().x;
    mouse_position_y = pt.y - this->GetScreenPosition().y;
    if (application_graph_active_id > -1 && application_graph_active_id < ags.size()) {
        application_graph_hovering_node(application_graph_active_id);
        if (move_node_started && application_graph_hovering_node_id > -1) {
            struct application_graph_node* current_node = ags[application_graph_active_id]->nodes[application_graph_hovering_node_id];
            current_node->pos_x += (mouse_position_x - mouse_down_mouse_x);
            current_node->pos_y += (mouse_position_y - mouse_down_mouse_y);
            mouse_down_mouse_x = mouse_position_x;
            mouse_down_mouse_y = mouse_position_y;
            myApp->drawPane->Refresh();
        }
    }
}

bool ctrl_pressed = false;
void BasicDrawPane::keyPressed(wxKeyEvent& event) {
    if (event.GetKeyCode() == wxKeyCode::WXK_CONTROL) {
        ctrl_pressed = true;
    }
}

void BasicDrawPane::keyReleased(wxKeyEvent& event) {
    if (event.GetKeyCode() == wxKeyCode::WXK_CONTROL) {
        ctrl_pressed = false;
    }
}

void BasicDrawPane::mouseDown(wxMouseEvent& event) {
    const wxPoint pt = wxGetMousePosition();
    mouse_down_mouse_x = pt.x - this->GetScreenPosition().x;
    mouse_down_mouse_y = pt.y - this->GetScreenPosition().y;
    if (application_graph_active_id > -1 && application_graph_active_id < ags.size() && application_graph_hovering_node_id > -1) {
        float dist_out = -1.0f;
        int closest_id = application_graph_is_on_input(application_graph_active_id, application_graph_hovering_node_id, mouse_down_mouse_x, mouse_down_mouse_y, &dist_out);
        if (dist_out == -1.0f || dist_out >= 6) {
            move_node_started = true;
        } else {
            if (ctrl_pressed) {
                if (closest_id > -1) {
                    application_graph_delete_edge(application_graph_active_id, application_graph_hovering_node_id, closest_id);
                }
            }
        }
    }
}

void BasicDrawPane::mouseReleased(wxMouseEvent& event) {
    if (!move_node_started) {
        if (application_graph_active_id > -1 && application_graph_active_id < ags.size()) {
            application_graph_add_edge(application_graph_active_id, mouse_down_mouse_x, mouse_down_mouse_y, mouse_position_x, mouse_position_y);
        }
    } else {
        move_node_started = false;
    }
    Refresh();
}

void BasicDrawPane::rightClick(wxMouseEvent& event) {
    wxString str;
    const wxPoint pt = wxGetMousePosition();
    right_click_mouse_x = pt.x - this->GetScreenPosition().x;
    right_click_mouse_y = pt.y - this->GetScreenPosition().y;
    BasicDrawPane::OnShowContextMenu(event);
}

void BasicDrawPane::OnShowContextMenu(wxMouseEvent& event) {
    wxMenu *menu;
    if (application_graph_hovering_node_id > -1) {
        menu = new wxMenu(wxT("Node Actions"));
        if (ags[application_graph_active_id]->nodes[application_graph_hovering_node_id]->process != nullptr) {
            menu->Append(MENU_ID_START_NODE, wxT("Start/Stop"));
        }
        menu->Append(MENU_ID_EDIT_NODE, wxT("Edit"));
        menu->Append(MENU_ID_EDIT_NODE_SETTINGS, wxT("Edit Settings"));
        menu->Append(MENU_ID_DELETE_NODE, wxT("Delete"));
    } else {
        menu = new wxMenu(wxT("Add Node"));
        menu->Append(MENU_ID_SHARED_MEMORY_BUFFER, wxT("Shared Memory Buffer"));
        menu->Append(MENU_ID_VIDEO_SOURCE, wxT("Video Source"));
        menu->Append(MENU_ID_IM_SHOW, wxT("Im Show"));
        menu->Append(MENU_ID_MASK_RCNN, wxT("Mask RCNN"));
        menu->Append(MENU_ID_GPU_VIDEO_ALPHA_MERGE, wxT("GPU Video Alpha Merge"));
        menu->Append(MENU_ID_GPU_MEMORY_BUFFER, wxT("GPU Memory Buffer"));
        menu->Append(MENU_ID_GPU_DENOISE, wxT("GPU Denoise"));
        menu->Append(MENU_ID_GPU_MOTION_BLUR, wxT("GPU Motion Blur"));
        menu->Append(MENU_ID_GPU_GAUSSIAN_BLUR, wxT("GPU Gaussian Blur"));
        menu->Append(MENU_ID_GPU_COMPOSER, wxT("GPU Composer"));
        menu->Append(MENU_ID_GPU_COMPOSER_ELEMENT, wxT("GPU Composer Element"));
    }
    PopupMenu(menu);
}

void BasicDrawPane::OnContextMenuSelected(wxCommandEvent& event) {
    switch (event.GetId()) {
    case MENU_ID_SHARED_MEMORY_BUFFER:
        ui_manager_show_frame(AGCT_SHARED_MEMORY_BUFFER, application_graph_active_id);
        break;
    case MENU_ID_VIDEO_SOURCE:
        ui_manager_show_frame(AGCT_VIDEO_SOURCE, application_graph_active_id);
        break;
    case MENU_ID_IM_SHOW:
        ui_manager_show_frame(AGCT_IM_SHOW, application_graph_active_id);
        break;
    case MENU_ID_MASK_RCNN:
        ui_manager_show_frame(AGCT_MASK_RCNN, application_graph_active_id);
        break;
    case MENU_ID_GPU_VIDEO_ALPHA_MERGE:
        ui_manager_show_frame(AGCT_GPU_VIDEO_ALPHA_MERGE, application_graph_active_id);
        break;
    case MENU_ID_GPU_MEMORY_BUFFER:
        ui_manager_show_frame(AGCT_GPU_MEMORY_BUFFER, application_graph_active_id);
        break;
    case MENU_ID_GPU_DENOISE:
        ui_manager_show_frame(AGCT_GPU_DENOISE, application_graph_active_id);
        break;
    case MENU_ID_GPU_COMPOSER:
        ui_manager_show_frame(AGCT_GPU_COMPOSER, application_graph_active_id);
        break;
    case MENU_ID_GPU_COMPOSER_ELEMENT:
        ui_manager_show_frame(AGCT_GPU_COMPOSER_ELEMENT, application_graph_active_id);
        break;
    case MENU_ID_GPU_MOTION_BLUR:
        ui_manager_show_frame(AGCT_GPU_MOTION_BLUR, application_graph_active_id);
        break;
    case MENU_ID_GPU_GAUSSIAN_BLUR:
        ui_manager_show_frame(AGCT_GPU_GAUSSIAN_BLUR, application_graph_active_id);
        break;
    case MENU_ID_START_NODE:
        application_graph_start_stop_node(application_graph_active_id, application_graph_hovering_node_id);
        break;
    case MENU_ID_EDIT_NODE: {
            struct application_graph_node* agn = ags[application_graph_active_id]->nodes[application_graph_hovering_node_id];
            switch (agn->component_type) {
                case AGCT_GPU_COMPOSER: {
                    ui_manager_show_frame(AGCT_GPU_COMPOSER, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
                case AGCT_GPU_COMPOSER_ELEMENT: {
                    ui_manager_show_frame(AGCT_GPU_COMPOSER_ELEMENT, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
                case AGCT_GPU_DENOISE: {
                    ui_manager_show_frame(AGCT_GPU_DENOISE, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
                case AGCT_GPU_MEMORY_BUFFER: {
                    ui_manager_show_frame(AGCT_GPU_MEMORY_BUFFER, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
                case AGCT_GPU_VIDEO_ALPHA_MERGE: {
                    ui_manager_show_frame(AGCT_GPU_VIDEO_ALPHA_MERGE, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
                case AGCT_IM_SHOW: {
                    ui_manager_show_frame(AGCT_IM_SHOW, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
                case AGCT_MASK_RCNN: {
                    ui_manager_show_frame(AGCT_MASK_RCNN, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
                case AGCT_SHARED_MEMORY_BUFFER: {
                    ui_manager_show_frame(AGCT_SHARED_MEMORY_BUFFER, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
                case AGCT_VIDEO_SOURCE: {
                    ui_manager_show_frame(AGCT_VIDEO_SOURCE, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
                case AGCT_GPU_MOTION_BLUR: {
                    ui_manager_show_frame(AGCT_GPU_MOTION_BLUR, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
                case AGCT_GPU_GAUSSIAN_BLUR: {
                    ui_manager_show_frame(AGCT_GPU_GAUSSIAN_BLUR, application_graph_active_id, application_graph_hovering_node_id);
                    break;
                }
            }
        }
        break;
    case MENU_ID_EDIT_NODE_SETTINGS: {
        ui_manager_show_frame(AGCT_ANY_NODE_SETTINGS, application_graph_active_id, application_graph_hovering_node_id);
        break;
    }
    case MENU_ID_DELETE_NODE: {
        application_graph_delete_node(application_graph_active_id, application_graph_hovering_node_id);
        break;
    }
    default:
        break;
    }
}

/*
 * Called by the system of by wxWidgets when the panel needs
 * to be redrawn. You can also trigger this call by
 * calling Refresh()/Update().
 */
void BasicDrawPane::paintEvent(wxPaintEvent& evt)
{
    wxPaintDC dc(this);
    render(dc);
}

/*
 * Alternatively, you can use a clientDC to paint on the panel
 * at any time. Using this generally does not free you from
 * catching paint events, since it is possible that e.g. the window
 * manager throws away your drawing when the window comes to the
 * background, and expects you will redraw it when the window comes
 * back (by sending a paint event).
 *
 * In most cases, this will not be needed at all; simply handling
 * paint events and calling Refresh() when a refresh is needed
 * will do the job.
 */
void BasicDrawPane::paintNow()
{
    wxClientDC dc(this);
    render(dc);
}

/*
 * Here we do the actual rendering. I put it in a separate
 * method so that it can work no matter what type of DC
 * (e.g. wxPaintDC or wxClientDC) is used.
 */
void BasicDrawPane::render(wxDC& dc) {
    if (application_graph_active_id > -1 && application_graph_active_id < ags.size()) {
        application_graph_draw_nodes(ags[application_graph_active_id], dc);
        application_graph_draw_edges(ags[application_graph_active_id], dc);
    }
}