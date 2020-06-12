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

#include "Logger.h"

MyApp *myApp;

IMPLEMENT_APP(MyApp)

bool MyApp::OnInit()
{
    cuda_stream_handler_init();

    myApp = this;

    struct application_graph* ag = new application_graph();
    ags.push_back(ag);

    wxBoxSizer* sizer = new wxBoxSizer(wxHORIZONTAL);
    frame = new wxFrame((wxFrame*)NULL, -1, wxT("VideoProcessor"), wxPoint(50, 50), wxSize(800, 600));

    drawPane = new BasicDrawPane((wxFrame*)frame);
    sizer->Add(drawPane, 1, wxEXPAND);

    frame->SetSizer(sizer);
    frame->SetAutoLayout(true);

    frame->Show();

    vs_frame = new VideoSourceFrame((wxWindow*)frame);
    smb_frame = new SharedMemoryBufferFrame((wxWindow*)frame);
    is_frame = new ImShowFrame((wxWindow*)frame);
    mrcnn_frame = new MaskRCNNFrame((wxWindow*)frame);
    vam_frame = new GPUVideoAlphaMergeFrame((wxWindow*)frame);
    gmb_frame = new GPUMemoryBufferFrame((wxWindow*)frame);
    gd_frame = new GPUDenoiseFrame((wxWindow*)frame);
    gc_frame = new GPUComposerFrame((wxWindow*)frame);
    gce_frame = new GPUComposerElementFrame((wxWindow*)frame);

    return true;
}

BEGIN_EVENT_TABLE(BasicDrawPane, wxPanel)
// some useful events
 EVT_MOTION(BasicDrawPane::mouseMoved)
 EVT_LEFT_DOWN(BasicDrawPane::mouseDown)
 EVT_LEFT_UP(BasicDrawPane::mouseReleased)
 EVT_RIGHT_DOWN(BasicDrawPane::rightClick)
    /*
 EVT_LEAVE_WINDOW(BasicDrawPane::mouseLeftWindow)
 EVT_KEY_DOWN(BasicDrawPane::keyPressed)
 EVT_KEY_UP(BasicDrawPane::keyReleased)
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
    application_graph_hovering_node(0);
    if (move_node_started && application_graph_hovering_node_id > -1) {
        struct application_graph_node* current_node = ags[0]->nodes[application_graph_hovering_node_id];
        current_node->pos_x += (mouse_position_x - mouse_down_mouse_x);
        current_node->pos_y += (mouse_position_y - mouse_down_mouse_y);
        mouse_down_mouse_x = mouse_position_x;
        mouse_down_mouse_y = mouse_position_y;
        myApp->drawPane->Refresh();
    }
}

void BasicDrawPane::mouseDown(wxMouseEvent& event) {
    const wxPoint pt = wxGetMousePosition();
    mouse_down_mouse_x = pt.x - this->GetScreenPosition().x;
    mouse_down_mouse_y = pt.y - this->GetScreenPosition().y;
    if (application_graph_hovering_node_id > -1) {
        float dist_out = -1.0f;
        application_graph_is_on_input(0, application_graph_hovering_node_id, mouse_down_mouse_x, mouse_down_mouse_y, &dist_out);
        if (dist_out == -1.0f || dist_out >= 6) {
            move_node_started = true;
        }
    }
}

void BasicDrawPane::mouseReleased(wxMouseEvent& event) {
    if (!move_node_started) {
        application_graph_add_edge(0, mouse_down_mouse_x, mouse_down_mouse_y, mouse_position_x, mouse_position_y);
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
        if (ags[0]->nodes[application_graph_hovering_node_id]->process != nullptr) {
            menu->Append(MENU_ID_START_NODE, wxT("Start/Stop Node"));
        }
    } else {
        menu = new wxMenu(wxT("Add Node"));
        menu->Append(MENU_ID_SHARED_MEMORY_BUFFER, wxT("Shared Memory Buffer"));
        menu->Append(MENU_ID_VIDEO_SOURCE, wxT("Video Source"));
        menu->Append(MENU_ID_IM_SHOW, wxT("Im Show"));
        menu->Append(MENU_ID_MASK_RCNN, wxT("Mask RCNN"));
        menu->Append(MENU_ID_GPU_VIDEO_ALPHA_MERGE, wxT("GPU Video Alpha Merge"));
        menu->Append(MENU_ID_GPU_MEMORY_BUFFER, wxT("GPU Memory Buffer"));
        menu->Append(MENU_ID_GPU_DENOISE, wxT("GPU Denoise"));
        menu->Append(MENU_ID_GPU_COMPOSER, wxT("GPU Composer"));
        menu->Append(MENU_ID_GPU_COMPOSER_ELEMENT, wxT("GPU Composer Element"));
    }
    PopupMenu(menu);
}

void BasicDrawPane::OnContextMenuSelected(wxCommandEvent& event) {
    switch (event.GetId()) {
    case MENU_ID_SHARED_MEMORY_BUFFER:
        myApp->smb_frame->Show(true);
        break;
    case MENU_ID_VIDEO_SOURCE:
        myApp->vs_frame->Show(true);
        break;
    case MENU_ID_IM_SHOW:
        myApp->is_frame->Show(true);
        break;
    case MENU_ID_MASK_RCNN:
        myApp->mrcnn_frame->Show(true);
        break;
    case MENU_ID_GPU_VIDEO_ALPHA_MERGE:
        myApp->vam_frame->Show(true);
        break;
    case MENU_ID_GPU_MEMORY_BUFFER:
        myApp->gmb_frame->Show(true);
        break;
    case MENU_ID_GPU_DENOISE:
        myApp->gd_frame->Show(true);
        break;
    case MENU_ID_GPU_COMPOSER:
        myApp->gc_frame->Show(true);
        break;
    case MENU_ID_GPU_COMPOSER_ELEMENT:
        myApp->gce_frame->Show(true);
        break;
    case MENU_ID_START_NODE:
        application_graph_start_stop_node(0, application_graph_hovering_node_id);
        break;
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
void BasicDrawPane::render(wxDC& dc)
{
    application_graph_draw_nodes(ags[0], dc);
    application_graph_draw_edges(ags[0], dc);

    // draw some text
    /*dc.DrawText(wxT("Testing"), 40, 60);

    // draw a circle
    dc.SetBrush(*wxGREEN_BRUSH); // green filling
    dc.SetPen(wxPen(wxColor(255, 0, 0), 5)); // 5-pixels-thick red outline
    dc.DrawCircle(wxPoint(200, 100), 25 /* radius *//*);

    // draw a rectangle
    dc.SetBrush(*wxBLUE_BRUSH); // blue filling
    dc.SetPen(wxPen(wxColor(255, 175, 175), 10)); // 10-pixels-thick pink outline
    dc.DrawRectangle(300, 100, 400, 200);

    // draw a line
    dc.SetPen(wxPen(wxColor(0, 0, 0), 3)); // black line, 3 pixels thick
    dc.DrawLine(300, 100, 700, 300); // draw line across the rectangle

    // Look at the wxDC docs to learn how to draw other stuff
    */
}